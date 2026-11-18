import sys

USAGE = """usage: python main.py {train,inference} [options]
       python main.py q4 {train,inference,benchmark} [options]

commands:
  train      train PointNet models (Q1 classification, Q2 segmentation)
             --task cls | seg | all   (all = cls then seg)
  inference  evaluate trained checkpoints; never trains
             --task cls | seg | all   (default all)
             --stage eval | q3 | all  (eval = Q1 for cls / Q2 for seg, q3 = robustness, all = eval then q3; default eval)

examples:
  python main.py train --task cls
  python main.py train --task all
  python main.py inference --task cls
  python main.py inference --stage q3
  python main.py inference --stage all

Q4 (Point Transformer, separate from Q1-Q3; see README.md):
  python main.py q4 benchmark --task cls|seg
  python main.py q4 train --task cls|seg|all
  python main.py q4 inference --task cls|seg|all

python main.py train -h and python main.py inference -h list every option."""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "train":
        import main_train
        main_train.main(rest, prog="main.py train")
    elif command == "inference":
        import main_inference
        main_inference.main(rest, prog="main.py inference")
    elif command == "q4" and rest[:1] in (["train"], ["benchmark"]):
        import train_q4
        train_q4.main(rest[1:], prog="main.py q4 " + rest[0], mode=rest[0])
    elif command == "q4" and rest[:1] == ["inference"]:
        import eval_q4
        eval_q4.main(rest[1:], prog="main.py q4 inference")
    else:
        print("unknown command '{}'\n\n{}".format(command, USAGE), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
