import argparse
import json
from .engine import build_index, search

def main():
    p = argparse.ArgumentParser(description='Service fleet GraphRAG')
    sub = p.add_subparsers(dest='command', required=True)
    index = sub.add_parser('index')
    index.add_argument('--input', default='data/incidents.json')
    index.add_argument('--db', default='fleet.sqlite')
    index.add_argument('--embedding-model', help='Optional sentence-transformers model')
    index.add_argument('--extract-model', help='Optional OpenAI-compatible extraction model')
    ask = sub.add_parser('ask')
    ask.add_argument('question')
    ask.add_argument('--db', default='fleet.sqlite')
    ask.add_argument('--tag', action='append', default=[])
    ask.add_argument('--mode', choices=['auto', 'local', 'global'], default='auto')
    ask.add_argument('--limit', type=int, default=5)
    args = p.parse_args()
    if args.command == 'index':
        extractor = None
        if args.extract_model:
            from .models import extractor_for
            extractor = extractor_for(args.extract_model)
        build_index(args.input, args.db, extractor, args.embedding_model)
        print('Index built successfully.')
    else:
        print(json.dumps(search(args.db, args.question, args.tag, args.mode, args.limit), indent=2))

if __name__ == '__main__': main()
