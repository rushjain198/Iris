#!/bin/bash
# Iris demo video: 45 s split screen, 1920x1080.
#   Left: third-person walk (portrait phone video).  Top right: Iris HUD screen recording + its audio.
#   Bottom right: rotating LiDAR scan of the course.  Bottom middle: title card.
#
#   ./compose_demo.sh WALK.mp4 HUD.mov SCAN.stl OUT.mp4
#
# Sync: the HUD recording runs longer than the walk, so it is cut at HUD_END (the moment the walk ends) and both
# clips are stretched to SECS. An optional span CUT0..CUT1 (HUD seconds) is removed from the HUD and the matching
# span from the walk, so both jump together and stay in sync.
#   HUD_END=67 CUT0=43.5 CUT1=47.6 SECS=45 ./compose_demo.sh ...      (defaults shown)
set -e
WALK=$1; HUD=$2; SCAN=$3; OUT=$4
[ -n "$OUT" ] || { echo "usage: $0 WALK.mp4 HUD.mov SCAN.stl OUT.mp4"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-$HERE/../.venv/bin/python}; [ -x "$PY" ] || PY=python3
T=${SECS:-45}; SR_END=${HUD_END:-67.0}; SR_C0=${CUT0:-43.5}; SR_C1=${CUT1:-47.6}
WORK=$(mktemp -d)
WA_END=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WALK")
K=$(echo "$WA_END/$SR_END" | bc -l)
WA_C0=$(echo "$SR_C0*$K" | bc -l); WA_C1=$(echo "$SR_C1*$K" | bc -l)
WA_LEN=$(echo "$WA_END-($WA_C1-$WA_C0)" | bc -l); SR_LEN=$(echo "$SR_END-($SR_C1-$SR_C0)" | bc -l)
FW=$(echo "$WA_LEN/$T" | bc -l); FS=$(echo "$SR_LEN/$T" | bc -l)
printf 'walk %.2f s, cut %.2f-%.2f, speed x%.3f | HUD to %.1f s, cut %.1f-%.1f, speed x%.3f\n' "$WA_END" "$WA_C0" "$WA_C1" "$FW" "$SR_END" "$SR_C0" "$SR_C1" "$FS"

echo "rendering LiDAR turntable..."; "$PY" "$HERE/lidar_turntable.py" "$SCAN" "$WORK/lidar"
"$PY" "$HERE/make_title.py" "$WORK/title.png" "$(printf %.1f "$FW")" "$(printf %.1f "$FS")"

ffmpeg -v error -y \
  -i "$WALK" -i "$HUD" -framerate 15 -i "$WORK/lidar/f%04d.png" -loop 1 -i "$WORK/title.png" \
  -filter_complex "
    color=c=0x05090e:s=1920x1080:r=30:d=$T[bg];
    [0:v]split[w1][w2];
    [w1]trim=0:$WA_C0,setpts=PTS-STARTPTS[wa];
    [w2]trim=$WA_C1:$WA_END,setpts=PTS-STARTPTS[wb];
    [wa][wb]concat=n=2:v=1:a=0,setpts=PTS/$FW,scale=608:1080:force_original_aspect_ratio=increase,crop=608:1080,fps=30[L];
    [1:v]split[s1][s2];
    [s1]trim=0:$SR_C0,setpts=PTS-STARTPTS[sa];
    [s2]trim=$SR_C1:$SR_END,setpts=PTS-STARTPTS[sb];
    [sa][sb]concat=n=2:v=1:a=0,setpts=PTS/$FS,scale=1312:720:force_original_aspect_ratio=decrease,pad=1312:720:(ow-iw)/2:(oh-ih)/2:color=0x05090e,fps=30[TR];
    [1:a]asplit[x1][x2];
    [x1]atrim=0:$SR_C0,asetpts=PTS-STARTPTS,afade=t=out:st=$(echo "$SR_C0-0.08" | bc -l):d=0.08[xa];
    [x2]atrim=$SR_C1:$SR_END,asetpts=PTS-STARTPTS,afade=t=in:d=0.08[xb];
    [xa][xb]concat=n=2:v=0:a=1,atempo=$FS,alimiter=limit=0.89,afade=t=in:d=0.3,afade=t=out:st=$(echo "$T-0.8" | bc -l):d=0.8[a];
    [2:v]fps=30,scale=640:360[LI];
    [3:v]scale=672:360,fps=30[TI];
    [bg][L]overlay=0:0[a1];
    [a1][TR]overlay=608:0[a2];
    [a2][TI]overlay=608:720[a3];
    [a3][LI]overlay=1280:720,drawbox=x=1280:y=720:w=640:h=360:color=0x5ff3ff@0.35:t=2,drawbox=x=608:y=0:w=1312:h=720:color=0x5ff3ff@0.25:t=2[v]" \
  -map "[v]" -map "[a]" -t "$T" -r 30 -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart "$OUT"
rm -rf "$WORK"
ffprobe -v error -show_entries format=duration:stream=codec_type,width,height -of compact=p=0 "$OUT"
