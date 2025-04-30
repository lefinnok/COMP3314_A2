{
  description = "Python development environment with python-lsp";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShell = pkgs.mkShell {
          name = "python-dev-shell";

          buildInputs = with pkgs; [
            # Python and related tools
            poetry
            python3
            python3Packages.python-lsp-server
            python3Packages.black
            python3Packages.flake8
            python3Packages.mypy
            python3Packages.pytest
            python3Packages.ipython
	    python3Packages.numpy
	    python3Packages.tkinter
	    python3Packages.torch
	    python3Packages.torchvision
	    python3Packages.matplotlib

            # Additional development utilities
            sqlite
            httpie
          ];

          shellHook = ''
            # Custom shell prompt
            export PS1="\[\033[1;36m\][python-dev-shell]\[\033[0m\] \[\033[1;32m\]\u@\h\[\033[0m\] \[\033[1;34m\]\w\[\033[0m\] \$ "

            # Create a virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment..."
              python -m venv .venv
            fi
            
            # Activate the virtual environment
            source .venv/bin/activate
            
            # Create a requirements.txt file if it doesn't exist
            if [ ! -f "requirements.txt" ]; then
              cat > requirements.txt << EOF
Flask==2.3.3
Flask-SQLAlchemy==3.1.1
Flask-SocketIO==5.3.4
python-socketio==5.8.0
python-engineio==4.5.1
flask-wtf==1.1.1
PyPDF2==3.0.1
EOF
              echo "Created requirements.txt with the required package versions."
            fi
            
            # Check if requirements need to be installed
            if [ ! -f ".venv/.installed" ] || [ requirements.txt -nt .venv/.installed ]; then
              echo "Installing requirements..."
              pip install --upgrade pip
              pip install -r requirements.txt
              touch .venv/.installed
            else
              echo "Requirements already installed."
            fi

            # Information about the environment
            echo "Python development environment activated."
            echo "Available tools: python, poetry, black, flake8, mypy, pytest, ipython, sqlite, httpie"
            echo "LSP server is available for editor integration."
            echo "Virtual environment is active with the specified package versions."
          '';
        };
      }
    );
}
