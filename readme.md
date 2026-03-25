# Personal Scripts

Little terminal utilties I mostly vibe-coded, maybe you'll find them handy too.

-------

# Pool Temp

Is the rooftop pool warm enough to swim in? Based on specific heat capacity of pool water volume to absorb last 5 days of local weather.

## Install

    pip3 install requests

## Run

    python3 pool_temp.py

--------

# Chroma Key

Input `,mp4` video and hex background color and generates transparent background `.webm` and `.mov` (for safari) files.

I use this to create the videos inside colored container sections on [kinopio.club/about](https://kinopio.club/about). After conversion, I use handbrake to compress the mov file.

## Install

    pip3 install opencv-contrib-python

## Run

    python3 chroma_key.py input.mp4 "#00FF00"

