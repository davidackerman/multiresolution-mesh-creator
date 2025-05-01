import sys
from multiresolution_mesh_creator.src.create_multiresolution_meshes import main
from pathlib import Path

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    sys.argv.append(str(root_dir / 'local-config'))
    sys.argv.append('-n')
    sys.argv.append('1')

    print("Command-line arguments:", sys.argv)

    main()
