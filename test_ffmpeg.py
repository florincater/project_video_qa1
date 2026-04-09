import subprocess
import shutil
import os

print("=== FFmpeg Debug Info ===\n")

# Check if ffmpeg is in PATH
ffmpeg_path = shutil.which('ffmpeg')
print(f"FFmpeg in PATH: {ffmpeg_path}")

# Check common locations
common_paths = [
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]

for path in common_paths:
    exists = os.path.exists(path)
    print(f"Exists at {path}: {exists}")

# Try to run ffmpeg
try:
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, 
                          text=True,
                          timeout=5)
    if result.returncode == 0:
        print("\n✅ FFmpeg is working!")
        print(result.stdout.split('\n')[0])
    else:
        print(f"\n❌ FFmpeg returned error code: {result.returncode}")
except FileNotFoundError:
    print("\n❌ FFmpeg command not found")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n=== Current PATH ===")
print(os.environ.get('PATH', 'PATH not found'))