from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import ClassVar, Generic, cast

from typing_extensions import TypeVar

from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import (
    AnyFloat,
    AnyFloatConstr,
    ArrayAttr,
    ArrayOfConstraint,
    ComplexType,
    FloatAttr,
    IntegerAttr,
    IntegerType,
)
from xdsl.interfaces import ConstantLikeInterface
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
)
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    EqIntConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    RangeOf,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

ComplexTypeConstr = ComplexType.constr(AnyFloat)

_ComplexNumberElementType = TypeVar(
    "_ComplexNumberElementType",
    bound=AnyFloat,
    covariant=True,
    default=AnyFloat,
)


@irdl_attr_definition
class ComplexNumberAttr(ParametrizedAttribute, Generic[_ComplexNumberElementType]):
    name = "complex.number"

    real: FloatAttr[_ComplexNumberElementType]
    imag: FloatAttr[_ComplexNumberElementType]
    complex_type: ComplexType[_ComplexNumberElementType]

    def __init__(
        self,
        real: FloatAttr[_ComplexNumberElementType],
        imag: FloatAttr[_ComplexNumberElementType],
        type: ComplexType[_ComplexNumberElementType],
    ):
        super().__init__(real, imag, type)

    @staticmethod
    def from_floats(value: tuple[float, float], type: AnyFloat) -> ComplexNumberAttr:
        return ComplexNumberAttr(
            FloatAttr(value[0], type), FloatAttr(value[1], type), ComplexType(type)
        )

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(
                f":{self.complex_type.element_type} {self.real.value.data}, {self.imag.value.data}"
            )
        printer.print_string(f" : {self.complex_type}")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """
        Example:
        ```mlir
        #complex.number<:f64 1.0, 2.0>
        #complex.number<:f64 1.0, 2.0> : complex<f64>
        ```
        """
        with parser.in_angle_brackets():
            parser.parse_punctuation(":")
            assert isa(element_type := parser.parse_type(), AnyFloat)
            real = FloatAttr(parser.parse_float(), element_type)
            parser.parse_punctuation(",")
            imag = FloatAttr(parser.parse_float(), element_type)
        if parser.parse_optional_punctuation(":"):
            assert isa(
                (complex_type := parser.parse_optional_attribute()),
                ComplexType[AnyFloat],
            )
            assert complex_type.element_type == element_type
        return [
            real,
            imag,
            ComplexType(element_type),
        ]


class ComplexUnaryComplexResultOperation(IRDLOperation, abc.ABC):
    """Base class for unary operations on complex numbers."""

    T: ClassVar = VarConstraint("T", ComplexTypeConstr)
    complex = operand_def(T)
    result = result_def(T)
    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))

    traits = traits_def(Pure())

    assembly_format = (
        "$complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)"
    )

    def __init__(
        self,
        operand: SSAValue | Operation,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        operand_ssa = SSAValue.get(operand)
        super().__init__(
            operands=[operand_ssa],
            result_types=[operand_ssa.type],
            properties={"fastmath": fastmath},
        )


class ComplexUnaryRealResultOperation(IRDLOperation, abc.ABC):
    """Base class for unary operations on complex numbers that return a float."""

    T: ClassVar = VarConstraint("T", AnyFloatConstr)

    complex = operand_def(ComplexType.constr(T))
    result = result_def(T)
    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))

    traits = traits_def(Pure())

    assembly_format = (
        "$complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)"
    )

    def __init__(
        self,
        operand: SSAValue | Operation,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        operand_ssa = SSAValue.get(operand)
        element_type = cast(ComplexType, operand_ssa.type).element_type
        super().__init__(
            operands=[operand_ssa],
            result_types=[element_type],
            properties={"fastmath": fastmath},
        )

    def verify_(self):
        element_type = cast(ComplexType, self.complex.type).element_type
        if self.result.type != element_type:
            raise VerifyException(
                f"result type {self.result.type} does not match "
                f"complex element type {element_type}"
            )


class ComplexBinaryOp(IRDLOperation, abc.ABC):
    """Base class for binary operations on complex numbers."""

    T: ClassVar = VarConstraint("T", ComplexTypeConstr)
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)
    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))

    traits = traits_def(Pure())

    assembly_format = (
        "$lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)"
    )

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        lhs_ssa = SSAValue.get(lhs)
        super().__init__(
            operands=[lhs_ssa, rhs],
            result_types=[lhs_ssa.type],
            properties={"fastmath": fastmath},
        )


class ComplexCompareOp(IRDLOperation, abc.ABC):
    """Base class for comparison operations on complex numbers."""

    T: ClassVar = VarConstraint("T", ComplexTypeConstr)
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(IntegerType(1))

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs)"

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        super().__init__(operands=[lhs, rhs], result_types=[IntegerType(1)])


@irdl_op_definition
class AbsOp(ComplexUnaryRealResultOperation):
    name = "complex.abs"


@irdl_op_definition
class AddOp(ComplexBinaryOp):
    name = "complex.add"


@irdl_op_definition
class AngleOp(ComplexUnaryRealResultOperation):
    name = "complex.angle"


@irdl_op_definition
class Atan2Op(ComplexBinaryOp):
    name = "complex.atan2"


@irdl_op_definition
class BitcastOp(IRDLOperation):
    name = "complex.bitcast"
    operand = operand_def()
    result = result_def()

    traits = traits_def(Pure())

    assembly_format = "$operand attr-dict `:` type($operand) `to` type($result)"

    def __init__(self, operand: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[operand], result_types=[result_type])


@irdl_op_definition
class ConjOp(ComplexUnaryComplexResultOperation):
    name = "complex.conj"


@irdl_op_definition
class ConstantOp(IRDLOperation, ConstantLikeInterface):
    name = "complex.constant"
    T: ClassVar = VarConstraint("T", AnyFloatConstr | base(IntegerType))
    value = prop_def(
        ArrayOfConstraint(
            RangeOf(
                AnyOf(
                    [
                        ParamAttrConstraint(IntegerAttr, (AnyAttr(), T)),
                        ParamAttrConstraint(FloatAttr, (AnyAttr(), T)),
                    ]
                )
            ).of_length(EqIntConstraint(2))
        )
    )

    # In contrast to other operations, `complex.constant` can
    # have any complex result type, not just floating point:
    complex = result_def(ComplexType.constr(T))

    traits = traits_def(Pure())

    assembly_format = "$value attr-dict `:` type($complex)"

    def __init__(self, value: ArrayAttr, result_type: ComplexType):
        super().__init__(properties={"value": value}, result_types=[result_type])

    def get_constant_value(self) -> Attribute:
        return self.value

    @staticmethod
    def from_floats(value: tuple[float, float], type: AnyFloat) -> ConstantOp:
        return ConstantOp(
            ArrayAttr([FloatAttr(value[0], type), FloatAttr(value[1], type)]),
            ComplexType(type),
        )

    @staticmethod
    def from_ints(value: tuple[int, int], type: IntegerType) -> ConstantOp:
        return ConstantOp(
            ArrayAttr([IntegerAttr(value[0], type), IntegerAttr(value[1], type)]),
            ComplexType(type),
        )


@irdl_op_definition
class CosOp(ComplexUnaryComplexResultOperation):
    name = "complex.cos"


@irdl_op_definition
class CreateOp(IRDLOperation):
    name = "complex.create"
    T: ClassVar = VarConstraint("T", AnyFloatConstr)
    real = operand_def(T)
    imaginary = operand_def(T)
    complex = result_def(ComplexType.constr(T))

    traits = traits_def(Pure())

    assembly_format = "$real `,` $imaginary attr-dict `:` type($complex)"

    def __init__(
        self,
        real: SSAValue | Operation,
        imaginary: SSAValue | Operation,
        result_type: ComplexType,
    ):
        super().__init__(operands=[real, imaginary], result_types=[result_type])


@irdl_op_definition
class DivOp(ComplexBinaryOp):
    name = "complex.div"


@irdl_op_definition
class EqualOp(ComplexCompareOp):
    name = "complex.eq"


@irdl_op_definition
class ExpOp(ComplexUnaryComplexResultOperation):
    name = "complex.exp"


@irdl_op_definition
class Expm1Op(ComplexUnaryComplexResultOperation):
    name = "complex.expm1"


@irdl_op_definition
class ImOp(ComplexUnaryRealResultOperation):
    name = "complex.im"


@irdl_op_definition
class LogOp(ComplexUnaryComplexResultOperation):
    name = "complex.log"


@irdl_op_definition
class Log1pOp(ComplexUnaryComplexResultOperation):
    name = "complex.log1p"


@irdl_op_definition
class MulOp(ComplexBinaryOp):
    name = "complex.mul"


@irdl_op_definition
class NegOp(ComplexUnaryComplexResultOperation):
    name = "complex.neg"


@irdl_op_definition
class NotEqualOp(ComplexCompareOp):
    name = "complex.neq"


@irdl_op_definition
class PowOp(ComplexBinaryOp):
    name = "complex.pow"


@irdl_op_definition
class ReOp(ComplexUnaryRealResultOperation):
    name = "complex.re"


@irdl_op_definition
class RsqrtOp(ComplexUnaryComplexResultOperation):
    name = "complex.rsqrt"


@irdl_op_definition
class SignOp(ComplexUnaryComplexResultOperation):
    name = "complex.sign"


@irdl_op_definition
class SinOp(ComplexUnaryComplexResultOperation):
    name = "complex.sin"


@irdl_op_definition
class SqrtOp(ComplexUnaryComplexResultOperation):
    name = "complex.sqrt"


@irdl_op_definition
class SubOp(ComplexBinaryOp):
    name = "complex.sub"


@irdl_op_definition
class TanOp(ComplexUnaryComplexResultOperation):
    name = "complex.tan"


@irdl_op_definition
class TanhOp(ComplexUnaryComplexResultOperation):
    name = "complex.tanh"


Complex = Dialect(
    "complex",
    [
        AbsOp,
        AddOp,
        AngleOp,
        Atan2Op,
        BitcastOp,
        ConjOp,
        ConstantOp,
        CosOp,
        CreateOp,
        DivOp,
        EqualOp,
        ExpOp,
        Expm1Op,
        ImOp,
        LogOp,
        Log1pOp,
        MulOp,
        NegOp,
        NotEqualOp,
        PowOp,
        ReOp,
        RsqrtOp,
        SignOp,
        SinOp,
        SqrtOp,
        SubOp,
        TanOp,
        TanhOp,
    ],
    [
        ComplexNumberAttr,
    ],
)
