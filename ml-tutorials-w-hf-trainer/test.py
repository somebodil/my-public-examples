import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foo', default='bar')
parser.set_defaults(foo='spam')
print(parser.parse_args())  # Namespace(foo='spam')
