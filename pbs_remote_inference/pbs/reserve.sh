#!/bin/bash
# Serve one robot's models every weekday 09:00-18:00 (Asia/Singapore) without cron. A PBS standing
# reservation keeps one GPU free for that window, and one cohost job per day waits in the
# reservation's queue (held with qsub -a until 08:55 that day). When a window ends, PBS deletes the
# running job; the next day's job starts at 09:00. Run on the login node, from the repo root:
#
#   bash pbs_remote_inference/pbs/reserve.sh <so101|yam> <last day YYYYMMDD>   # new reservation + jobs
#   bash pbs_remote_inference/pbs/reserve.sh <so101|yam> --resv <S-id>         # jobs for an existing one
#
# --resv only adds the days that have no queued job, so it is safe to rerun. DRY_RUN=1 prints the
# qsub commands instead. Check with pbs_rstat and qstat -wu $USER; stop with pbs_rdel <S-id>, which
# also deletes its queued jobs. Endpoints as in README.md: colosseum-so101:9101-9105 and
# colosseum-yam:9111-9115. Job output: logs/pbs_remote_inference/reserved/<robot>-<YYYYMMDD>.log
set -e
cd "$(dirname "$0")/../.."    # repo root
export LC_ALL=C PBS_TZID=${PBS_TZID:-Asia/Singapore}
USAGE="usage: reserve.sh <so101|yam> <last day YYYYMMDD>  |  reserve.sh <so101|yam> --resv <S-id>"
ROBOT=${1:?$USAGE}
case "$ROBOT" in   # MODELS order fixes the ports, which the relays rely on (README.md)
    so101) MODELS=${MODELS:-molmoact2:pi05:groot:g05:lingbot} ;;
    yam)   MODELS=${MODELS:-molmoact2:pi05:groot:lingbot:g05} ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
START=0900 END=1800 BYDAY=MO,TU,WE,TH,FR
RES=select=1:ncpus=12:ngpus=1:mem=225gb   # what a 1-GPU cohost job gets on Hopper
WALLTIME=08:59:00                         # fits inside the 9 h window
RUN=${DRY_RUN:+echo}
attr() { awk -F' = ' -v k="$1" '{ sub(/^[ \t]+/, "", $1) } $1 == k { print $2; exit }' <<< "$INFO"; }

if [ "${2:-}" = --resv ]; then
    RESV=${3:?$USAGE}
else
    LAST=${2:?$USAGE}
    [[ "$LAST" =~ ^[0-9]{8}$ ]] || { echo "$USAGE" >&2; exit 1; }
    # First occurrence: today if 09:00 is still ahead and today is a listed day, else the next one.
    d=0; [ "$(date +%H%M)" -lt "$START" ] || d=1
    until [[ ",$BYDAY," == *",$(date -d "+$d day" +%a | cut -c1-2 | tr a-z A-Z),"* ]]; do d=$((d + 1)); done
    FIRST=$(date -d "+$d day" +%Y%m%d)
    RESV=$(pbs_rsub -N "colosseum-$ROBOT" -R "$FIRST$START" -E "$FIRST$END" \
        -r "FREQ=WEEKLY;BYDAY=$BYDAY;UNTIL=${LAST}T235959" -l "$RES" | awk '{print $1}')
    echo "Requested $RESV ($FIRST to $LAST); waiting for PBS to confirm it..."
    state=""
    for _ in $(seq 60); do
        INFO=$(pbs_rstat -f "$RESV" 2>/dev/null || true)
        state=$(attr reserve_state)
        [ "$state" = RESV_CONFIRMED ] && break
        [ -n "$state" ] || { echo "ERROR: $RESV was denied (no GPU free for every occurrence?)" >&2; exit 1; }
        sleep 5
    done
    [ "$state" = RESV_CONFIRMED ] || { echo "ERROR: $RESV is still $state after 5 min; delete it with pbs_rdel $RESV" >&2; exit 1; }
fi

INFO=$(pbs_rstat -f "$RESV")
QUEUE=$(attr queue)
NEXT=$(attr reserve_start)                                       # current or next occurrence
LEFT=$(( $(attr reserve_count) - $(attr reserve_index) + 1 ))    # occurrences from NEXT on
RBYDAY=$(attr reserve_rrule | grep -oE 'BYDAY=[A-Z,]+' | cut -d= -f2)
[ -n "$QUEUE" ] && [ "$LEFT" -gt 0 ] && [ -n "$RBYDAY" ] || { echo "ERROR: cannot read $RESV:"; echo "$INFO"; exit 1; } >&2
echo "$RESV (queue $QUEUE, nodes $(attr resv_nodes)): $LEFT occurrences from $NEXT"

QUEUED=" $(qstat -wu "$USER" 2>/dev/null | awk -v q="$QUEUE" '$3 == q { printf "%s ", $4 }')"
mkdir -p logs/pbs_remote_inference/reserved
day=$(date -d "$NEXT" +%Y-%m-%d); n=0; added=0
while [ "$n" -lt "$LEFT" ]; do
    if [[ ",$RBYDAY," == *",$(date -d "$day" +%a | cut -c1-2 | tr a-z A-Z),"* ]]; then
        D=$(date -d "$day" +%Y%m%d); NAME=$ROBOT-${D:4:4}; n=$((n + 1))
        if [[ "$QUEUED" == *" $NAME "* ]]; then
            echo "  $NAME already queued"
        else
            $RUN qsub -q "$QUEUE" -N "$NAME" -a "${D}0855" -l "$RES" -l walltime=$WALLTIME \
                -o "logs/pbs_remote_inference/reserved/$ROBOT-$D.log" \
                -v "EMBODIMENT=$ROBOT,MODELS=$MODELS,TUNNEL=tailscale" pbs_remote_inference/pbs/cohost.pbs
            added=$((added + 1))
        fi
    fi
    day=$(date -d "$day + 1 day" +%Y-%m-%d)
done
echo "Queued $added job(s) in $QUEUE."
