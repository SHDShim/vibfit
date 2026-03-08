def main():
    # import triggers application startup
    from . import app  # noqa: F401
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
