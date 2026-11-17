# Sourced by the PBS scripts: the first model port for each embodiment. The same ranges as NSCC
# (nscc_pbs_remote_inference/pbs/env.sh), so nscc_pbs_remote_inference/run_nscc_eval.sh's relays
# also reach Hopper jobs, and an so101 and a yam job on the same node don't collide on 127.0.0.1.
base_port() {  # base_port <robot>
    case "$1" in
        so101)  echo 9101 ;;
        yam)    echo 9111 ;;
        franka) echo 9121 ;;
        *) echo "ERROR: unknown embodiment '$1' (so101, yam, franka)" >&2; return 1 ;;
    esac
}
