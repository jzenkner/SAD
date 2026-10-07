"""Operations for the DeepCoder domain."""
from multiprocessing.managers import Token
# pylint: disable=unidiomatic-typecheck
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import functools
import re
import ast
import copy

from absl import flags
from absl import app

from tasks import operation_base
from tasks.value import Value, Variable, ConstantValue, InputVariable, OperationValue, get_bound_variable, \
  BoundVariable, FreeVariable
from tasks.operation_base import ParseError, LambdaBeamError, RunError


_MAX_CONST = flags.DEFINE_integer(
    'max_const', 5,
    'The maximum value of a LambdaBeam integer constant.')

def lambdabeam_max_list_length() -> int:
  """Hides the internal flag from code outside this module."""
  return operation_base.MAX_LIST_LENGTH.value


def lambdabeam_max_int() -> int:
  """Hides the internal flag from code outside this module."""
  return operation_base.MAX_INT.value


def lambdabeam_min_int() -> int:
  # Originally LambdaBeam used the range [-256, 255] where the endpoints are
  # asymmetric. We use symmetric endpoints here.
  return -1 * lambdabeam_max_int()


def lambdabeam_max_const() -> int:
  """Hides the internal flag from code outside this module."""
  return _MAX_CONST.value


def lambdabeam_min_const() -> int:
  return -1 * lambdabeam_max_const()


def lambdabeam_small_value_filter(x):
  """Checks whether a single value is ok for LambdaBeam."""
  if x is None:
    return False
  if isinstance(x, int):
    return -256 <= x <= 255
  if isinstance(x, list):
    return (len(x) <= 20 and
            # isinstance(False, int) is True. We don't want booleans in lists.
            all(type(e) is int  # pylint: disable=unidiomatic-typecheck
                and lambdabeam_small_value_filter(e)
                for e in x))
  return True


# Types for type specifications, e.g., `([int], bool)` has type LambdaType and
# specifies a lambda that takes 1 int and returns a bool.
InputsType = List[Union[Type[Any], 'LambdaType']]
OutputType = Type[Any]
LambdaType = Tuple[InputsType, OutputType]

# A type for the result of applying any operation.
ResultType = Union[int, bool, List[int], List[bool]]


class Example(object):
  """A DeepCoder specification in the form of an I/O example."""

  def __init__(self, inputs: List[ResultType], output: ResultType):
    self.inputs = inputs
    self.output = output


class ProgramTask(object):
  """A DeepCoder program with I/O examples."""

  def __init__(self, program: 'Program', examples: List[Example]):
    self.program = program
    self.examples = examples


def join_token_lists(token_lists: List[List[str]],
                     separator_token: str) -> List[str]:
  return functools.reduce(lambda a, b: a + [separator_token] + b, token_lists)


def validate_int(i: int) -> bool:
  """Checks that the integer is in range."""
  return lambdabeam_min_int() <= i <= lambdabeam_max_int()


def validate_result(result: ResultType) -> bool:
  """Returns whether an object is valid or not."""
  # Distinguish between bool and int.
  # pylint: disable=unidiomatic-typecheck
  if type(result) == int:
    return validate_int(result)
  elif type(result) == list:
    return all(type(x) == int for x in result) and all(validate_int(x) for x in result)
  else:
    return False
  # pylint: enable=unidiomatic-typecheck


def tokenize_result(result: ResultType) -> List[str]:
  """Returns a list of tokens for the result of an operation."""
  if isinstance(result, (int, bool)):
    return [str(result)]
  elif isinstance(result, list):
    return ['['] + sum([tokenize_result(x) for x in result], []) + [']']
  elif isinstance(result, Value):
    return result.tokenized_expression()
  else:
    raise operation_base.LambdaBeamError(f'Unhandled type in tokenize_result({result})')


def result_to_str(result: ResultType) -> str:
  return ' '.join(tokenize_result(result))


def str_to_result(result_str: str) -> Optional[ResultType]:
    """
    Parse a result string into a concrete Python object.

    Handles:
    - ints, e.g. "3" -> 3
    - bools, e.g. "True" -> True
    - lists of ints or bools, e.g. "[1 2 3]" -> [1, 2, 3] or "[True False True]" -> [True False True]
    """
    result_str = result_str.strip()

    # Fix comma-less lists: "[1 2 3]" -> "[1, 2, 3]"
    # Works for ints and booleans
    result_str = re.sub(r'(?<=\d) +(?=-?\d)', ', ', result_str)
    result_str = re.sub(r'(?<=True) +(?=True|False)', ', ', result_str)
    result_str = re.sub(r'(?<=False) +(?=True|False)', ', ', result_str)

    # Try literal evaluation
    try:
        val = ast.literal_eval(result_str)
        # Validate type
        if isinstance(val, (int, bool)):
            return val
        elif isinstance(val, list) and all(isinstance(x, (int, bool)) for x in val):
            return val
        else:
            return None
    except (ValueError, SyntaxError):
        # Handle single boolean literals manually
        if result_str in ("True", "False"):
            return result_str == "True"
        return None


def variable_token(index: int) -> str:
  return f'x{index}'

MAX_NUM_VARIABLES = 10  # How many variables are available to use.
ALL_VARIABLES = frozenset(variable_token(index)
                          for index in range(MAX_NUM_VARIABLES))


def variable_index_from_token(token: str) -> int:
  if not re.fullmatch(r'x\d+', token) and not re.fullmatch(r'v\d+', token):
    raise operation_base.ParseError(f'Invalid variable token: {token}')
  index = int(token[1:])
  if index < 0 or index >= MAX_NUM_VARIABLES:
    raise operation_base.ParseError(f'Variable token has out-of-bounds index: {token}')
  return index


class ProgramState(object):
  """Holds a program state (for one example)."""

  def __init__(self, state: List[ResultType], variables: List[str]):
    self.state = list(state)
    self.variables = list(variables)
    if len(state) != len(variables):
      raise operation_base.LambdaBeamError(
          f'`state` has length {len(state)} but `variables` has length '
          f'{len(variables)}')
    self._variable_to_result = {token: result
                                for token, result in zip(variables, state)}

  def __len__(self) -> int:
    return len(self.state)

  def __bool__(self) -> bool:
    return all(self.state)

  def get_index(self, index: int) -> ResultType:
    if index < 0 or index >= len(self.state):
      raise operation_base.RunError(f'Invalid index: {index}')
    return self.state[index]

  def get_variable(self, token: str) -> ResultType:
    if token not in self._variable_to_result:
      raise operation_base.RunError(f'Invalid variable token: {token}')
    return self._variable_to_result[token]

  def __eq__(self, other: Any) -> bool:
    return (isinstance(other, ProgramState) and
            self.state == other.state and
            self.variables == other.variables)

  def copy(self) -> 'ProgramState':
    return ProgramState(list(self.state), list(self.variables))

  def add_result(self, result: ResultType, variable: str) -> None:
    if variable in self._variable_to_result:
      raise operation_base.RunError(f'Cannot add new result for variable {variable} that '
                     f'already exists in state: {self}')
    self.state.append(result)
    self.variables.append(variable)
    self._variable_to_result[variable] = result

  def get_output(self) -> ResultType:
    return self.state[-1]

  def get_output_variable(self) -> str:
    return self.variables[-1]

  def tokenize(self) -> List[str]:
    lines = [[token, '='] + tokenize_result(result)
             for token, result in zip(self.variables, self.state)]
    return join_token_lists(lines, separator_token='|')

  def get_context(self) -> InputVariable:
    return InputVariable([self.state[-1]], self.variables[-1])


  def __str__(self) -> str:
    return ' '.join(self.tokenize())

  @classmethod
  def from_tokens(cls, tokens: List[str]) -> 'ProgramState':
    return cls.from_str(' '.join(tokens))

  @classmethod
  def from_str(cls, string: str) -> 'ProgramState':
    """Creates a ProgramState from its string representation."""
    lines = [line.strip() for line in string.split('|')]
    state = []
    variables = []
    for line in lines:
      splitted = line.split('=')
      if len(splitted) != 2:
        raise operation_base.ParseError(f"Expected exactly one '=': {line}")
      lhs, rhs = [part.strip() for part in splitted]
      _ = variable_index_from_token(lhs)  # Make sure lhs has the right format.
      variable = lhs
      if variable in variables:
        raise operation_base.ParseError(f'Found duplicate variable: {variable}')
      result = str_to_result(rhs)
      result_is_valid = validate_result(result)
      if not result_is_valid:
        raise operation_base.ParseError(f'Found invalid result `{result}` from RHS {rhs!r}')
      state.append(result)
      variables.append(variable)
    return cls(state, variables)


def is_int(s: str) -> bool:
  try:
    int(s)
    return True
  except ValueError:
    return False


def extract_lambda_args(tokens):
  """
  Extracts lambda arguments and outer arguments from tokens without parentheses.

  Args:
      tokens: list of str, e.g. ['x3', '=', 'Map', 'Add', '1', 'x0']
      TOKEN_TO_LAMBDA: dict[str, callable] — map of lambda ops, e.g. {'Add': fn}
      input_vars: set[str] — input variable names, e.g. {'x0', 'x1'}
      LAMBDA_ARITY: dict[str, int] — maps each lambda op to its argument count, e.g. {'Add': 2}

  Returns:
      outer_args: list[str]
      lambda_args: list[str]
  """
  outer_args = []
  lambda_args = []

  i = 0
  while i < len(tokens):
    tok = tokens[i]

    # Case 1: lambda operation found
    if tok in TOKEN_TO_LAMBDA:
      arity = TOKEN_TO_LAMBDA[tok].arity

      # Peek at next tokens to see if constants follow
      next_tokens = tokens[i + 1: i + 1 + arity]

      # Infer how many arguments are variables vs constants
      largs = []
      num_constants = 0
      for nix, nt in enumerate(next_tokens):
        if isinstance(nt, InputVariable) and num_constants != 0 and len(next_tokens) == 2 and outer_args[-1] == 'ZipWith':
          # treat as free variable (v1, v2, …)
          continue
        elif is_int(nt) or (num_constants == 0 and len(next_tokens) == 2 and outer_args[-1] != 'ZipWith' and isinstance(nt, Value)):
          largs.append(nt)
          num_constants += 1

      # Fill in missing lambda params with v1, v2, … up to arity
      # The variables always come first
      for j in range(arity - num_constants):
        largs.append(f"v{j+1}")

      if arity - num_constants == 1:
        # this is only *not* the case for Zipwith and Scanl1 and thus for parameterized
        # lambdas like Subtract 1 which should yield lambda u1: u1 - 1 and not lambda u1: 1 - u1
        largs.reverse()
      lambda_args.append(largs)
      outer_args.append(tok)

      i += num_constants # skip tokens added to lambda args
    else:
      outer_args.append(tok)

    i += 1
  return outer_args, lambda_args


class Statement(object):
  def __init__(self, variable: str, operation: operation_base.OperationBase, args: list):
    self.variable = variable
    self.operation = operation  # Now a LambdaBeam OperationBase
    self.args = args  # Could be Variables or Lambdas
    self.contains_int_variable = False
    for arg in self.args:
      if isinstance(arg, OperationValue):
        if any(isinstance(aarg, ConstantValue) and type(aarg.constant) == str for aarg in arg.arg_values):
          self.contains_int_variable = True

  def run(self, initial_state: ProgramState) -> Optional[ProgramState]:
    # Prevent mutating the original parsed statement
    if self.contains_int_variable:
      self = copy.deepcopy(self)
    resolved_args = []
    arg_variables = []

    for arg in self.args:
        if isinstance(arg, Variable):
            val = initial_state.get_variable(arg.name)
            resolved_args.append(InputVariable([val], arg.name))
            arg_variables.append([])
        elif isinstance(arg, OperationValue) and arg.free_variables:
            if self.operation.token != 'ZipWith':
              for c, ele in enumerate(arg.arg_values):
                if isinstance(ele, ConstantValue) and isinstance(ele.constant, str) and ele.constant.startswith('x'):
                  arg.arg_values[c] = ConstantValue(initial_state.get_variable(ele.constant))
                  arg = arg.operation.apply(arg.arg_values, free_variables=arg.free_variables)
                  break

            # It's a lambda! Need to bind its free vars.
            bound_vars = [BoundVariable(v.name.replace('v', 'u')) for v in arg.free_variables]
            resolved_args.append(arg)
            arg_variables.append(bound_vars)
        elif isinstance(arg, str) and arg.startswith('x'):
          val = initial_state.get_variable(arg)
          resolved_args.append(InputVariable([val], arg))
          arg_variables.append([])
        else:
          resolved_args.append(arg)
          arg_variables.append(getattr(arg, 'free_variables', []))

    # Apply the operation
    result = self.operation.apply(
        arg_values=resolved_args,
        arg_variables=arg_variables
    )

    if not result:
      return None

    result_state = initial_state.copy()
    result_state.add_result(result.values[0], self.variable)

    return result_state

  def tokenize(self) -> List[str]:
    """Tokenize the statement for serialization."""
    tokens = [self.variable, "=", self.operation.name]

    for arg in self.args:
      if isinstance(arg, Variable):
        tokens.append(arg.name)
      elif isinstance(arg, OperationValue):
        tokens.append(repr(arg.operation))
        if arg.contains_lambda:
          tokens +=  [str(ele.constant) for ele in arg.arg_values if isinstance(ele, ConstantValue)]
      elif isinstance(arg, FirstOrderLambdaBeamOperation):
        tokens += [repr(arg)]
      elif isinstance(arg, ConstantValue):
        tokens += [str(arg.constant)]
      else:
        tokens += [str(arg)]
    return tokens

  def __str__(self):
    return " ".join(self.tokenize())

  @classmethod
  def from_tokens(cls, tokens: List[str],
                  check_variable_name: bool = True) -> 'Statement':
    """Parses a Statement from a list of tokens."""
    # Get params for higher-order and lambda functions
    outer_args, lambda_args = extract_lambda_args(tokens)

    # Parse LHS variable, =, and the operation.
    if len(outer_args) < 4:
      raise operation_base.ParseError(f'Too few tokens: {outer_args}')
    variable = outer_args[0]
    if check_variable_name:
      _ = variable_index_from_token(variable)  # Check its format.
    if outer_args[1] != '=':
      raise operation_base.ParseError(f"Second token must be '=': {outer_args}")
    operation_token = outer_args[2]
    if operation_token not in TOKEN_TO_OPERATION:
      raise operation_base.ParseError(f'Unknown operation {operation_token} in: {outer_args}')
    operation = TOKEN_TO_OPERATION[operation_token]

    # Parse operation arguments.
    args = []
    for i, token in enumerate(outer_args[3:]):
      if token in TOKEN_TO_LAMBDA:
        if not operation.has_any_bound_variables or i > 0:
          raise operation_base.ParseError(f'Did not expect lambda at token {i+3}: {outer_args}')
        op = TOKEN_TO_LAMBDA[token]
        largs = lambda_args.pop(0)

        free_vars = []
        for lidx, larg in enumerate(largs):
          if larg.startswith('v'): # free variable
            larg = FreeVariable(larg)
            largs[lidx] = larg
            free_vars.append(larg)
          elif larg.startswith('u'): # bound variable
            largs[lidx] = BoundVariable(larg)
          elif is_int(larg):
            larg = int(larg)
            largs[lidx] = ConstantValue(larg)
          else:
            largs[lidx] = ConstantValue(larg)
        arg = op.apply(largs, free_variables=free_vars)
      else:
        if operation.has_any_bound_variables and i == 0:
          raise operation_base.ParseError(f'Expected lambda for token {i+3}: {tokens}')
        arg = token
        if arg == variable:
          raise operation_base.ParseError(f'Cannot use LHSe variable as arg: {tokens}')
        try:
          arg = int(arg)
          arg = ConstantValue(arg)
        except ValueError:
          _ = variable_index_from_token(arg)  # Check its format.
      args.append(arg)

    if len(args) != operation.arity:
      raise operation_base.ParseError(
          f'Statement tokens have wrong arity {operation.arity} for operation: {tokens}')

    return cls(variable, operation, args)

  @classmethod
  def from_str(cls, string: str, check_variable_name: bool = True) -> 'Statement':
    return cls.from_tokens(string.split(' '), check_variable_name)


class Program(object):
  def __init__(self, input_variables: List[str], statements: List[Statement]):
    self.input_variables = list(input_variables)
    self.statements = list(statements)
    self.num_inputs = len(input_variables)

    if (len(set(self.get_variables()))
        != len(statements) + len(input_variables)):
      raise operation_base.RunError(
          f'A variable token is duplicated in the program with '
          f'input_variables={input_variables} and statements={statements}.')

  def get_variables(self) -> List[str]:
    return self.input_variables + [s.variable for s in self.statements]

  def run(self, inputs: List[ResultType]) -> Optional[ProgramState]:
    if len(inputs) != self.num_inputs:
      raise operation_base.RunError(f'Got {len(inputs)} inputs but expected {self.num_inputs}')
    state = ProgramState(inputs, self.input_variables)
    for statement in self.statements:
      state = statement.run(state)
      if state is None:
        return None
    return state

  def __len__(self) -> int:
    return len(self.statements)

  def tokenize(self) -> List[str]:
    lines = []
    for input_variable in self.input_variables:
      lines.append([input_variable, '=', 'INPUT'])
    for statement in self.statements:
      lines.append(statement.tokenize())
    return join_token_lists(lines, separator_token='|')

  def __str__(self) -> str:
    return ' '.join(self.tokenize())

  @classmethod
  def from_tokens(cls, tokens: List[str]) -> 'Program':
    """Parses a Program from a list of tokens."""
    # Split tokens into lines at '|'.
    lines = []
    current_line = []
    for token in tokens:
      if token == '|':
        lines.append(current_line)
        current_line = []
      else:
        current_line.append(token)
    if current_line:
      lines.append(current_line)

    # Parse statements.
    found_non_input = False
    input_variables = []
    statements = []
    for line in lines:
      if not line:
        raise operation_base.ParseError('Encoutered empty line')
      if line[-1] == 'INPUT':
        if found_non_input:
          raise operation_base.ParseError(f'Found INPUT after a statement: {lines}')
        variable = line[0]
        _ = variable_index_from_token(variable)  # Check its format.
        if line != [variable, '=', 'INPUT']:
          raise operation_base.ParseError(f'Error while parsing INPUT statement: {line}')
        input_variables.append(variable)
      else:
        found_non_input = True
        statements.append(Statement.from_tokens(line))
    return cls(input_variables, statements)

  @classmethod
  def from_str(cls, string: str) -> 'Program':
    return cls.from_tokens(string.split(' '))


class FirstOrderLambdaBeamOperation(operation_base.OperationBase):
  """A base class for DeepCoder operations."""

  def __init__(self, inputs_type=None, output_type=None, *args, **kwargs):
    super(FirstOrderLambdaBeamOperation, self).__init__(self.__class__.__name__,
                                             *args, **kwargs)
    self.token = self.name
    self.func = self.apply_single
    self.inputs_type = inputs_type
    self.output_type = output_type

  def run(self, inputs: List[Any]) -> Optional[ResultType]:
    """Runs an operation on input arguments."""
    if len(inputs) != self.arity:
      raise operation_base.RunError(f'Arity was {len(inputs)} but expected {self.arity}')
    try:
      result = self.func(inputs)
    except TypeError as e:
      raise operation_base.RunError(
          f'Encountered TypeError when running {self.func} on {inputs}') from e
    if not validate_result(result):
      return None
    return result


class LambdaOperation(FirstOrderLambdaBeamOperation):
  def __init__(self, inputs_type=None, output_type=None, *args, **kwargs):
    super(FirstOrderLambdaBeamOperation, self).__init__(self.__class__.__name__,
                                                        *args, **kwargs)
    self.token = self.name
    self.func = self.apply_single
    self.inputs_type = inputs_type
    self.output_type = output_type


class HigherOrderLambdaBeamOperation(operation_base.OperationBase):
  """A base class for DeepCoder operations."""

  def __init__(self, inputs_type=None, output_type=None, *args, **kwargs):
    super(HigherOrderLambdaBeamOperation, self).__init__(self.__class__.__name__,
                                             *args, **kwargs)
    self.token = self.name
    self.name = self.token
    self.func = self.apply_single
    self.inputs_type = inputs_type
    self.output_type = output_type

  def run(self, inputs: List[Any]) -> Optional[ResultType]:
    """Runs an operation on input arguments."""
    if len(inputs) != self.arity:
      raise operation_base.RunError(f'Arity was {len(inputs)} but expected {self.arity}')
    try:
      state = ProgramState(inputs[1].values, ['x0'])
      if len(inputs) > 2:
        state.add_result(inputs[2].values[0], 'x1')
      s = Statement('x9', self, inputs)
      result = s.run(state)
      result = result.get_output()
    except TypeError as e:
      raise operation_base.RunError(
          f'Encountered TypeError when running {self.func} on {inputs}') from e
    if not validate_result(result):
      return None
    return result
################################################################################
# First-order functions returning int.
################################################################################


class Add(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Add, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) not in (int, str) or type(right) not in (int, str):
      return None
    return left + right


class Subtract(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Subtract, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left - right


class Multiply(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Multiply, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left * right


class IntDivide(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(IntDivide, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left // right


class Square(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Square, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:
      return None
    return x ** 2


class Min(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Min, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return min(left, right)


class Max(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Max, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return max(left, right)


################################################################################
# First-order functions taking or returning bool.
################################################################################


class Greater(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Greater, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left > right


class Less(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Less, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left < right


class Equal(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(Equal, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) not in [int, bool] or type(left) != type(right):
      return None
    return left == right


class IsEven(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(IsEven, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:
      return None
    return x % 2 == 0


class IsOdd(LambdaOperation):

  def __init__(self, inputs_type, output_type):
    super(IsOdd, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:
      return None
    return x % 2 == 1


################################################################################
# First-order functions manipulating lists (returning list or an element).
################################################################################


class Head(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Head, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if not x or len(x) == 0:
      return None
    return x[0]


class Last(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Last, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if not x or len(x) == 0:
      return None
    return x[-1]


class Take(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Take, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    n, xs = raw_args
    if type(n) is not int:
      return None
    return xs[:n]


class Drop(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Drop, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    n, xs = raw_args
    if type(n) is not int or not xs:
      return None
    return xs[n:]


class Access(FirstOrderLambdaBeamOperation):
  """DeepCoder's Access operation."""

  def __init__(self, inputs_type, output_type):
    super(Access, self).__init__(arity=2, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    n, xs = raw_args
    # DeepCoder chooses to error if n is negative; we use Python's negative
    # indexing convention (our DSL is a superset of DeepCoder's anyway).
    if type(n) is not int or n >= len(xs):
      return None
    return xs[n]


class Minimum(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Minimum, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    if len(xs) == 0:
      return None
    return min(xs)


class Maximum(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Maximum, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    if len(xs) == 0:
      return None
    return max(xs)


class Reverse(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Reverse, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return list(reversed(xs))


class Sort(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Sort, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return sorted(xs)


class Sum(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Sum, self).__init__(arity=1, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return sum(xs)


################################################################################
# Higher-order functions.
################################################################################


class Map(HigherOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(Map, self).__init__(arity=2, num_bound_variables=[1, 0], inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    f, xs = raw_args
    return list(map(f, xs))


class Filter(HigherOrderLambdaBeamOperation):
  """DeepCoder's Filter operation."""

  def __init__(self, inputs_type, output_type):
    super(Filter, self).__init__(arity=2, num_bound_variables=[1, 0], inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    f, xs = raw_args
    conditions = [f(x) for x in xs]
    if not all(isinstance(x, bool) for x in conditions):
      return None
    return [x for x, c in zip(xs, conditions) if c]


class Count(HigherOrderLambdaBeamOperation):
  """DeepCoder's Count operation."""

  def __init__(self, inputs_type, output_type):
    super(Count, self).__init__(arity=2, num_bound_variables=[1, 0], inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    f, xs = raw_args
    conditions = [f(x) for x in xs]
    if not all(isinstance(x, bool) for x in conditions):
      return None
    return sum(conditions)


class ZipWith(HigherOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(ZipWith, self).__init__(arity=3, num_bound_variables=[2, 0, 0], inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    f, xs, ys = raw_args
    return [f(x, y) for x, y in zip(xs, ys)]


class Scanl1(HigherOrderLambdaBeamOperation):
  """DeepCoder's Scanl1 operation."""

  def __init__(self, inputs_type, output_type):
    super(Scanl1, self).__init__(arity=2, num_bound_variables=[2, 0], inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    f, xs = raw_args
    ys = []
    for i, x in enumerate(xs):
      if i == 0:
        ys.append(x)
      else:
        ys.append(f(ys[-1], x))
    return ys

"""class If(FirstOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(If, self).__init__(arity=3, inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    condition, x, y = raw_args
    if type(x) == int:
      x = [x] * len(condition)
    if type(y) == int:
      y = [y] * len(condition)

    res = []
    for c, xi, yi in zip(condition, x, y):
      if type(c) is not bool or type(xi) is not int or type(yi) is not int:
        return None
      res.append(xi if c else yi)
    return res"""

class If(HigherOrderLambdaBeamOperation):

  def __init__(self, inputs_type, output_type):
    super(If, self).__init__(arity=4, num_bound_variables=[1, 0, 0, 0], inputs_type=inputs_type, output_type=output_type)

  def apply_single(self, raw_args):
    condition, condition_ipt, if_out, else_out = raw_args

    res = []
    for c, xi, yi in zip(condition_ipt, if_out, else_out):
      cond_eval = condition(c)
      if type(cond_eval) is not bool or type(xi) is not int or type(yi) is not int:
        return None
      res.append(xi if cond_eval else yi)
    return res

def get_operations():
  return list(TOKEN_TO_OPERATION.values())

# token to lambda is contains all operations but first-order operations that cannot be used inside a lambda, e.g., Head(l)
TOKEN_TO_OPERATION = {
  'Add': Add([int, int], int),
  'Subtract': Subtract([int, int], int),
  'Multiply': Multiply([int, int], int),
  'IntDivide': IntDivide([int, int], int),
  'Square': Square([int], int),
  'Min': Min([int, int], int),
  'Max': Max([int, int], int),
  'Greater': Greater([int, int], bool),
  'Less': Less([int, int], bool),
  'Equal': Equal([int, int], bool),
  'IsEven': IsEven([int], bool),
  'IsOdd': IsOdd([int], bool),
  'Head': Head([list], int),
  'Last': Last([list], int),
  'Take': Take([int, list], list),
  'Drop': Drop([int, list], list),
  'Access': Access([int, list], int),
  'Minimum': Minimum([list], int),
  'Maximum': Maximum([list], int),
  'Reverse': Reverse([list], list),
  'Sort': Sort([list], list),
  'Sum': Sum([list], int),
  'Map': Map([([int, int], int), list], list),
  'Filter': Filter([([int], bool), list], list),
  'Count': Count([([int], bool), list], int),
  'ZipWith': ZipWith([([int, int], int), list, list], list),
  'Scanl1': Scanl1([([int, int], int), list], list),
  'If': If([([int, int], bool), list, list, list], list)
}

FIRST_ORDER_OPERATIONS = {
  'Head': Head([list], int),
  'Last': Last([list], int),
  'Take': Take([int, list], list),
  'Drop': Drop([int, list], list),
  'Access': Access([int, list], int),
  'Minimum': Minimum([list], int),
  'Maximum': Maximum([list], int),
  'Reverse': Reverse([list], list),
  'Sort': Sort([list], list),
  'Sum': Sum([list], int)
}

HIGHER_ORDER_OPERATIONS = {
  'Map': Map([([int, int], int), list], list),
  'Filter': Filter([([int], bool), list], list),
  'Count': Count([([int], bool), list], int),
  'ZipWith': ZipWith([([int, int], int), list, list], list),
  'Scanl1': Scanl1([([int, int], int), list], list),
  'If': If([([int, int], bool), list, list, list], list)
}

TOKEN_TO_LAMBDA = {
  'Add': Add([int, int],int),
  'Subtract': Subtract([int, int], int),
  'Multiply': Multiply([int, int], int),
  'IntDivide': IntDivide([int, int], int),
  'Min': Min([int, int], int),
  'Max': Max([int, int], int),
  'Greater': Greater([int, int], bool),
  'Less': Less([int, int], bool),
  'Equal': Equal([int, int], bool),
  'IsEven': IsEven([int], bool),
  'IsOdd': IsOdd([int], bool),
  'Square': Square([int], int),
}

# Subsets of DSL functionality for generating compositional generalization
# datasets.
LAMBDAS_ONLY_MINUS_MIN = [TOKEN_TO_LAMBDA['Subtract'], TOKEN_TO_LAMBDA['Min']]
LAMBDAS_ONLY_PLUS_MUL_MAX = [TOKEN_TO_LAMBDA['Add'], TOKEN_TO_LAMBDA['Multiply'], TOKEN_TO_LAMBDA['Max']]
OPERATIONS_ONLY_SCAN = [TOKEN_TO_OPERATION['Scanl1']]
OPERATIONS_NO_SCAN = [op for tok, op in TOKEN_TO_OPERATION.items() if tok != 'Scanl1']
FIRST_ORDER_AND_MAP = list(FIRST_ORDER_OPERATIONS.values()) + [TOKEN_TO_OPERATION['Map']]
HIGHER_ORDER_NO_MAP = [op for tok, op in HIGHER_ORDER_OPERATIONS.items()
                       if tok != 'Map']

PAD, BOS, EOS, SEP = '', '<BOS>', '<EOS>', '|'
PAD_ID, BOS_ID, EOS_ID, SEP_ID = 0, 1, 2, 3

def vocab_tables() -> Tuple[Dict[int, str], Dict[str, int]]:
  """Returns id-to-token and token-to-id vocabulary mappings."""
  # These tokens should be constant unless we change the DSL.
  tokens = [PAD, BOS, EOS, SEP]
  tokens.extend(['=', 'INPUT', '[', ']'])
  tokens.extend(op.token for op in get_operations())

  # These tokens may change based on flags or other settings.
  tokens.extend(variable_token(index) for index in range(MAX_NUM_VARIABLES))
  tokens.extend(str(i)
                for i in range(lambdabeam_min_int(), lambdabeam_max_int()+1))
  # tokens.extend(['False', 'True'])

  # Construct mappings between tokens and ids.
  id_to_token = {i: token for i, token in enumerate(tokens)}
  token_to_id = {token: i for i, token in enumerate(tokens)}
  assert len(id_to_token) == len(token_to_id)
  return id_to_token, token_to_id

def main(_):
  # [[4], [-22], 4], [[3, 0, 3], [-42, 30, -17], 3]], [[3, 2], [-26, 28, -26], 3]
  #state = ProgramState([[0, 2], [-22, -11], 2], ['x9', 'x0', 'x1'])
  """for ps_st in ['x9 = [ 4 ] | x0 = [ -22 ] | x1 = 4',  'x9 = [ 3 0 3 ] | x0 = [ -42 30 -17 ] | x1 = 3',  'x9 = [ 3 2 ] | x0 = [ -26 28 -26 ] | x1 = 3',  'x9 = [ 0 2 ] | x0 = [ -22 -11 ] | x1 = 2']:
    state = ProgramState.from_str(ps_st)
    print(str(state))
    s = Statement.from_str("x2 = Map Add x1 x9")
    print(s)
    print(s.run(state))"""
  
  """state = ProgramState([[10, 3, 21, 22, -3], [1, 2, 3, 4, 5]], ['x0', 'x1'])
  s = Statement.from_str("x2 = ZipWith Add x1 x0")
  print(s)
  print(s.run(state))"""
  s = Statement.from_str("x1 = Map Multiply Reverse x9")
  print(s)
  """
  p = Program.from_str("x0 = INPUT | x1 = INPUT | x2 = Access x1 x0 | x3 = Map Subtract x2 x0")
  #p = Statement.from_str("x9 = INPUT | x0 = INPUT | x1 = INPUT | x2 = Map Add x1 x9")
  ipts = [[25, 74, 34, 42, 60], 3]
  print(p)
  res = p.run(ipts)
  print(res)"""

  """
  'x9 = [ 3 0 3 ] | x0 = [ -42 30 -17 ] | x1 = 3', 'x9 = [ 3 2 ] | x0 = [ -26 28 -26 ] | x1 = 3', 'x9 = [ 0 2 ] | x0 = [ -22 -11 ] | x1 = 2']

  state = ProgramState([[11, -13, 45], [12, 3, -5]], ['x0', 'x1'])
  print(state)

  s = Statement.from_str('x5 = ZipWith Add 0 x1')
  print(s)

  res = s.run(state)
  print(res)
  assert res.get_output() == [23, -10, 40]"""

if __name__ == '__main__':
  app.run(main)