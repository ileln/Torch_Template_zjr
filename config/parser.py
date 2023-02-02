import argparse

parser = argparse.ArgumentParser(description='Arguments for running the scripts')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--exp_name', type=str, default="h36m", help="h36m / cmu")
parser.add_argument('--input_n', type=int, default=10, help="")
parser.add_argument('--output_n', type=int, default=25, help="")
parser.add_argument('--dct_n', type=int, default=35, help="")
parser.add_argument('--device', type=str, default="cuda:0", help="")
parser.add_argument('--num_works', type=int, default=0)
# parser.add_argument('--train_manner', type=str, default="all", help="all / 8")
parser.add_argument('--test_manner', type=str, action="append")
parser.add_argument('--train_manner', type=str, action="append")
# parser.add_argument('--train_manner', type=str)
parser.add_argument('--debug_step', type=int, default=1, help="")
parser.add_argument('--is_train', type=bool, default=True, help="")
parser.add_argument('--is_load', type=bool, default='', help="")
parser.add_argument('--model_path', type=str, default="", help="")

args = parser.parse_args()

if __name__ == "__main__":
    print(args.input_n)