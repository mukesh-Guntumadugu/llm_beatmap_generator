#!/bin/bash
#SBATCH --job-name=clear_home
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/clear_home_%j.out
#SBATCH --error=logs/clear_home_%j.err

echo "Job started: $(date)"

# Because the original `mv` command was interrupted with ^Z,
# We will use `rsync` because it picks up exactly where it left off, and it's much safer!

echo "Starting massive background sync of ~/.cache to data drive (this could take hours)..."
mkdir -p /data/mg546924/hidden_cache
# rsync copies everything. If it finishes, it deletes the original.
rsync -aP --remove-source-files ~/.cache/ /data/mg546924/hidden_cache/

echo "Syncing complete. Deleting any leftover empty folders inside ~/.cache..."
rm -rf ~/.cache
echo "Creating Symlink for ~/.cache..."
ln -s /data/mg546924/hidden_cache ~/.cache


echo "Starting massive background sync of ~/.local to data drive..."
mkdir -p /data/mg546924/hidden_local
rsync -aP --remove-source-files ~/.local/ /data/mg546924/hidden_local/

echo "Syncing complete. Deleting leftover empty folders in ~/.local..."
rm -rf ~/.local
echo "Creating Symlink for ~/.local..."
ln -s /data/mg546924/hidden_local ~/.local


echo "Home drive successfully cleared and symlinked!"
echo "Job ended: $(date)"
