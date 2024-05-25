for script in *.sh; do
    case "$script" in
        *vasso_metrics_1*|*vasso_metrics_2*|*vasso_metrics_3*|*vasso_k=5_rand*|*vasso_k=10_rand*|*vassore_k=2_rand*|*sbatch_all_scripts*)
            ;;
        *)
            sbatch "$script"
            ;;
    esac
done
