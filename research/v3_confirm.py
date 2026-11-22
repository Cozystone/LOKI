"""Run the preregistered confirmation without replacing the development results."""
import argparse
from . import v3_dark, v3_heat, v3_retention, v3_rooms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test", choices=("dark", "rooms", "retention", "heat"))
    name = parser.parse_args().test
    filename = "v3_confirm_" + name
    if name == "dark":
        v3_dark.run(range(3501,3513), filename, lattice=True)
    elif name == "rooms":
        v3_rooms.run(range(3601,3613), filename, lattice=True)
    elif name == "retention":
        v3_retention.run(range(3701,3713), filename)
    else:
        v3_heat.run(range(3801,3809), filename)


if __name__ == "__main__":
    main()
