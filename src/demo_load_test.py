from pathlib import Path

from test_loader import load_test


def main():
    # Υποθέτουμε ότι τρέχεις το script από το root του project (BiasMind/)
    project_root = Path(__file__).resolve().parents[1]
    test_path = project_root / "data" / "tests" / "test_bfi10.json"

    test_def = load_test(test_path)

    print(f"Test name: {test_def.test_name}")
    print(f"Language: {test_def.language}")
    print(f"Scale: {test_def.scale_min}–{test_def.scale_max}")
    print(f"Traits: {', '.join(test_def.traits)}")
    print(f"Number of items: {len(test_def.items)}")

    print("\nFirst item example:")
    first = test_def.items[0]
    print(f"  id={first.id}")
    print(f"  text={first.text}")
    print(f"  trait={first.trait}")
    print(f"  reverse={first.reverse}")


if __name__ == "__main__":
    main()
