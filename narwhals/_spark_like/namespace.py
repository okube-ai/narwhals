from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

from narwhals._expression_parsing import combine_root_names
from narwhals._expression_parsing import parse_into_exprs
from narwhals._expression_parsing import reduce_output_names
from narwhals._spark_like.expr import SparkLikeExpr
from narwhals._spark_like.utils import get_column_name
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self
    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.typing import IntoSparkLikeExpr
    from narwhals.utils import Version


class SparkLikeNamespace(CompliantNamespace["Column"]):
    def __init__(self, *, backend_version: tuple[int, ...], version: Version) -> None:
        self._backend_version = backend_version
        self._version = version

    def all(self) -> SparkLikeExpr:
        def _all(df: SparkLikeLazyFrame) -> list[Column]:
            import pyspark.sql.functions as F  # noqa: N812

            return [F.col(col_name) for col_name in df.columns]

        return SparkLikeExpr(  # type: ignore[abstract]
            call=_all,
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def all_horizontal(self, *exprs: IntoSparkLikeExpr) -> SparkLikeExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [reduce(operator.and_, cols).alias(col_name)]

        return SparkLikeExpr(  # type: ignore[abstract]
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="all_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def col(self, *column_names: str) -> SparkLikeExpr:
        return SparkLikeExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def when(
        self: Self,
        *predicates: IntoSparkLikeExpr,
    ) -> SparkLikeWhen:
        plx = self.__class__(backend_version=self._backend_version, version=self._version)
        if predicates:
            condition = plx.all_horizontal(*predicates)
        else:
            msg = "at least one predicate needs to be provided"
            raise TypeError(msg)

        return SparkLikeWhen(condition, self._backend_version, version=self._version)


class SparkLikeWhen:
    def __init__(
        self,
        condition: SparkLikeExpr,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherwise_value: Any = None,
        *,
        version: Version,
    ) -> None:
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value
        self._version = version

    def __call__(self: Self, df: ArrowDataFrame) -> Sequence[ArrowSeries]:
        import pyarrow as pa
        import pyarrow.compute as pc

        from narwhals._expression_parsing import parse_into_expr

        plx = df.__narwhals_namespace__()
        condition = parse_into_expr(self._condition, namespace=plx)(df)[0]
        try:
            value_series = parse_into_expr(self._then_value, namespace=plx)(df)[0]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            value_series = condition.__class__._from_iterable(
                pa.repeat(pa.scalar(self._then_value), len(condition)),
                name="literal",
                backend_version=self._backend_version,
                version=self._version,
            )

        value_series_native = value_series._native_series
        condition_native = condition._native_series

        if self._otherwise_value is None:
            otherwise_native = pa.repeat(
                pa.scalar(None, type=value_series_native.type), len(condition_native)
            )
            return [
                value_series._from_native_series(
                    pc.if_else(condition_native, value_series_native, otherwise_native)
                )
            ]
        try:
            otherwise_expr = parse_into_expr(self._otherwise_value, namespace=plx)
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression.
            # Remark that string values _are_ converted into expressions!
            return [
                value_series._from_native_series(
                    pc.if_else(
                        condition_native, value_series_native, self._otherwise_value
                    )
                )
            ]
        else:
            otherwise_series = otherwise_expr(df)[0]
            condition_native, otherwise_native = broadcast_series(
                [condition, otherwise_series]
            )
            return [
                value_series._from_native_series(
                    pc.if_else(condition_native, value_series_native, otherwise_native)
                )
            ]

    def then(self: Self, value: SparkLikeExpr | ArrowSeries | Any) -> SparkLikeThen:
        self._then_value = value

        return SparkLikeThen(
            self,
            depth=0,
            function_name="whenthen",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"value": value},
        )


class SparkLikeThen(SparkLikeExpr):
    def __init__(
        self: Self,
        call: SparkLikeWhen,
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._kwargs = kwargs

    def otherwise(self: Self, value: SparkLikeExpr | ArrowSeries | Any) -> ArrowExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `PandasWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self