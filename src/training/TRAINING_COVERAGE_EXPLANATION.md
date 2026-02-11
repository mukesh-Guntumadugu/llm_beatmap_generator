# Training Data Coverage Explanation

## Your Confusion (I understand!)
You see logs like:
- Batch 1: Processing Window [0:300]
- Batch 2: Processing Window [2400:2700]
- Batch 3: Processing Window [4800:5100]
- Batch 4: Processing Window [7200:7500]

And you think: "It's skipping from 300 to 2400! It's not processing the entire file!"

## The Truth: It IS Processing Everything!

### Here's What's Actually Happening:

**Your Audio File:**
- Total Frames: **9,679 frames**
- This is the ENTIRE "Springtime" song

**Sliding Window Settings:**
- Window Size: 300 frames (4 seconds of audio)
- Stride: 75 frames (move forward 1 second each time)

### How Many Samples Are Created?

The sliding window creates 126 samples:
- Sample 0: Frames [0:300]
- Sample 1: Frames [75:375]  ← Notice: Overlaps with Sample 0!
- Sample 2: Frames [150:450]
- Sample 3: Frames [225:525]
- ...
- Sample 125: Frames [9375:9679] (padded to 300)

**Total: 126 samples covering ALL 9,679 frames!**

### With Batch Size 32:

- **Batch 0**: Samples 0-31 (32 samples)
- **Batch 1**: Samples 32-63 (32 samples)
- **Batch 2**: Samples 64-95 (32 samples)
- **Batch 3**: Samples 96-125 (30 samples)

**Total: 4 batches × ~32 samples = 126 samples**

### Why the Logs Look Confusing:

The log only shows **Sample 0 from each batch**:
- Batch 0, Sample 0: Frame 0 (= 0 × 75)
- Batch 1, Sample 0: Frame 2400 (= 32 × 75)
- Batch 2, Sample 0: Frame 4800 (= 64 × 75)
- Batch 3, Sample 0: Frame 7200 (= 96 × 75)

**But each batch contains 32 hidden samples!**

For example, Batch 1 actually processes:
- Sample 32: Frames [2400:2700]
- Sample 33: Frames [2475:2775]
- Sample 34: Frames [2550:2850]
- ...
- Sample 63:Frames [4725:5025]

## Proof It Works:

Look at the log:
```
Total Samples Processed: 126
```

**126 samples × 75 frame stride = 9,450 frames covered**

Plus window size (300) = Coverage up to frame 9,750 ✓

**The ENTIRE 9,679 frame file is processed!**

## Bottom Line:

✅ **YES** - The model sees the entire audio file  
✅ **YES** - All 9,679 frames are processed  
✅ **YES** - The sliding window works correctly  

The logs just don't show all 126 samples individually because that would be too much output!
