# basic utility function
from typing import Any, Callable
import torch


def recursive_lambda(
    x: Any,
    _lambda: Callable,
    target_type: type = torch.Tensor,
    ignore_rest: bool = False,
):
    """
    torch.Tensor を含む、[list, tuple, dict]構造に対して
    再帰的に処理を行う。
    Args:
        x: torch.Tensor を含む、[list, tuple, dict]構造
        _lambda: target_type に対して行う処理メソッド
        target_type:(型 か tuple(型1, 型2, ...)) この型/これらの型に対して_lambda処理します
                strとか、 (str, int, float)とかも可
                この型がlist, tuple か dictの場合は_lambda処理してその中のを再帰しません。
        ignore_rest:(bool) Falseならlist,tuple,dictとtarget_typeかNoneのみを許可します。それ以外の型が出るとエラーを出します。
                Trueならそれ以外の型があっても何も操作せずに続きます。
    """
    T = type(x)
    if isinstance(x, target_type):
        return _lambda(x)
    elif isinstance(x, dict):
        return T(
            {
                k: recursive_lambda(
                    v, _lambda, target_type=target_type, ignore_rest=ignore_rest
                )
                for k, v in x.items()
            }
        )
    elif isinstance(x, (list, tuple)):
        return T(
            [
                recursive_lambda(
                    v, _lambda, target_type=target_type, ignore_rest=ignore_rest
                )
                for v in x
            ]
        )
    elif x is None:
        return None
    elif ignore_rest:
        return x
    else:
        raise TypeError(
            f"type of x must be in [{target_type}, dict, list, tuple, None] or its subclass, got {T.__name__}."
        )
