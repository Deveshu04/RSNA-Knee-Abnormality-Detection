import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "download", "-d", "wheels", "pylibjpeg", "pylibjpeg-libjpeg",
                "pylibjpeg-openjpeg", "python-gdcm"], check=True)
