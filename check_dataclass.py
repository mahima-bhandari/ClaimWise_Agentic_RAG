
import sys
import dataclasses
from dataclasses import dataclass, field

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Dataclasses module:", dataclasses.__file__)


@dataclass(slots=True)
class Simple:
    value: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Example:
    path: str
    _cache: str | None = field(
        default=None,
        init=False,
        repr=False,
    )


simple = Simple(value="test")
example = Example(path="test.py")

print("\nSimple dataclass slots:", Simple.__slots__)
print("Simple value:", simple.value)

print("\nExample dataclass slots:", Example.__slots__)
print("Example fields:", dataclasses.fields(Example))
print("Example cache exists:", hasattr(example, "_cache"))
print("Example object dictionary:", getattr(example, "__dict__", "No __dict__"))
