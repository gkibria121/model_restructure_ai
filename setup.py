import subprocess,os,sys
def setup_colab():
    """Helper function to set up Google Colab environment"""

    print("Setting up Google Colab environment...")

    # Install system-level packages
    try:
        subprocess.check_call(['apt-get', 'update'])
        subprocess.check_call(['apt-get', 'install', '-y', 'ffmpeg'])
        print("✓ Installed ffmpeg")
    except subprocess.CalledProcessError:
        print("✗ Failed to install ffmpeg")

    # Install required Python packages
    packages = [
        'librosa',
        'soundfile',
        'torchaudio',
        'seaborn',
        'tqdm',
        'demucs'
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

    # Create directory structure
    os.makedirs('./dataset/ai', exist_ok=True)
    os.makedirs('./dataset/real', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    print("✓ Created directory structure")
    print("✓ Colab setup complete!")
    print("\nNext steps:")
    print("1. Upload your AI audio files to ./dataset/ai/")
    print("2. Upload your Real audio files to ./dataset/real/")
    print("3. Run main() to start the application")

setup_colab()
print("Current working directory:", os.getcwd())