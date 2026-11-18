from train import create_parser, run


def main(argv=None, prog=None):
    run(create_parser(prog).parse_args(argv))


if __name__ == '__main__':
    main()
