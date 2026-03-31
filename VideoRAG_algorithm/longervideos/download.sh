#!/bin/bash

set -ex

courses="0-fights-in-animal-kingdom 1-nature-scenes 2-climate-week-at-columbia-engineering 3-black-myth-wukong 4-rag-lecture 5-ai-agent-lecture 6-daubechies-wavelet-lecture 7-daubechies-art-and-mathematics-lecture 8-tech-ceo-lecture 9-dspy-lecture 10-trading-for-beginners 11-primetime-emmy-awards 12-journey-through-china 13-fia-awards 14-education-united-nations 15-game-awards 16-ahp-superdecision 17-decision-making-science 18-elon-musk 19-jeff-bezos 20-12-days-of-openai 21-autogen"

for course in $courses; do
    mkdir -p ./$course/videos
    yt-dlp -o "%(id)s.%(ext)s" -S "res:720" -a "./$course/videos.txt" -P "./$course/videos" >> ./download_log.txt 2>&1
    wait
done