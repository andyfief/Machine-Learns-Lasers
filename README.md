## This is my current project! ##
I'm a big EDM fan, and I thought it would be cool to make a DMX laser controller with python. The goal is to create AI-assisted laser shows with my laser module.

## Goals ##
  Have a user interface, hopefully something native
  Let a machine learning model predict the current song "state" at any given moment (vocals, drop, pre-drop, buildup, etc)
    Or let it predict frames containing transitions between the two and fill in the gaps with logic. (Why not a generative solution? See Limitations)
  Use a predefined library of laser functions, controlled by these predicted labels.

## Limitations ##
I want temporal dependencies in the predictions
  
Labeling songs takes a long time
  
I have a humble amount of compute locally
  
  A generative solution producing raw dmx signals is hard for a couple reasons:
  - DMX signals have a discrete nature (for most channels) - values 254 and 255 might be completely different patterns, speeds, or directions.
  - A slight lapse in accuracy could mean a completely different pattern being generated than was "intended"
  - It would take a very long time to label songs with DMX channels myself (and I've never actually made a laser show before)
  - I'm labeling songs myself, with a labeling tool (see **Labeling Tool** below). I'm trying my best to be fair across sessions.
    
The shows will only be as good as the patterns I create.

I don't know a lot about music.

## Current Progress ## 
I've spent some time making the pre-defined patterns with dmx signals and a tkinter labeling tool. I've began labeling songs with speeds at a given frame, and pattern groups (see "Data")

One model will classify speed at each frame of a song, and another will classify patterns (or transitions). The combination of the two will hopefully result in a cool laser show.

I've also spent some time thinking about model architectures and appraoches.

## Data ##
The dataset I'm creating is a pair of continuous values, labeled at 10fps, for a song's duration.

One array labels the speed of the song at any moment. I label these myself using my ears and my own opinion, not something algorithmic. (I've found that this is much harder than it sounds but am open to revisiting it)

The other array represnents the pattern labels I assign each frame. The categories im using: 'Ambient, Vocals, Buildup, Buildup2, Pre-Drop, Drop, Drop2, Hold, and None.
<img width="1855" height="137" alt="image" src="https://github.com/user-attachments/assets/b3319833-db38-449c-96d0-0422cb01c5f7" />
<img width="1859" height="157" alt="image" src="https://github.com/user-attachments/assets/4f9822b1-4c83-4e34-afa1-7bc6c5a2ef75" />
  
I will also include MFCC values. MFCCs are values that describe the shape of a sound’s spectrum in a way that mimics human hearing.

The speed label/prediction will likely be used as input for the pattern prediction.

I'll also add columns like "time elapsed" "% of song completed"

## Labeling Tool ##
Most of the time spent on this project so far has been crafting an efficient labeling tool.
It is located in ```root/labeling/app/tk.py```

<img width="1920" height="1025" alt="image" src="https://github.com/user-attachments/assets/dfe59953-0f9d-4308-bc4a-d971ba13ecae" />
How it works:

  - User toggles an integer label value (0-9 for speeds, 0-8 for patterns)
    
  - While the song plays, each frame (sampled at 10fps) is labeled with the current value.
    
  - User can insert "dividers" that can be moved frame-by-frame.
    
  - User can set labels for entire sections between dividers.
    
  - Labeling for speed and patterns is done on one interface.
    
  - Patterns are saved independently, so you can come back and modify either array later
    
  - User can transfer 1 array -> the other, using "Patterns->Speed" or "Speed->Patterns" buttons. Useful because a lot of changes in one array happen on the same frames as the other.
    
  - User can right click a "plateau" to change the label value. A plateau is a section between any two rising/falling edges
    
- tk_withStem is deprecated
  
**You'll notice a lot of magic numbers in files unrelated to the labeling tool. DMX signals take in hardware-specific values in a 0-255 range for hardware-specific channels.** 
I've labeled a good portion of the dmx.set_channel calls in *some* files, but I'm not going to label everything in each file - If you're really curious as to what they're doing, you can check out the manual of the laser that im using, posted in the root dir.
Id like to make things more readable in the future but the pattern library isn't something I'm giving a lot of attention to right now.

