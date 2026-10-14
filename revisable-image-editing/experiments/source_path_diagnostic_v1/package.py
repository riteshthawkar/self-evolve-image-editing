"""Build a small standalone transfer archive; never include datasets/weights/secrets."""
from pathlib import Path
import argparse
import tarfile
from source_path.common import PACKAGE, sha

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    target=Path(args.output).resolve()
    if target.exists(): raise FileExistsError('Transfer archive already exists')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(target,'w:gz') as archive:
        for path in sorted(PACKAGE.rglob('*')):
            if path.is_symlink(): raise ValueError('Portable package must not contain symlinks')
            if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc': continue
            if path.suffix not in ('.py','.yaml','.json','.sh','.txt','.md'): continue
            archive.add(path,arcname=Path('source_path_diagnostic_v1')/path.relative_to(PACKAGE),recursive=False)
    print(str(target)); print('SHA256',sha(target))

if __name__=='__main__': main()
