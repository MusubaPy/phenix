# xvfb-run -s "-screen 0 1280x1024x24" \
env MJPC_CSV_LOG=/phenix_ws/mujoco_mpc/logs/quadruped_mod_alex_log.csv timeout 30s \
/phenix_ws/mujoco_mpc/build/bin/mjpc_mod --task "Quadruped Flat (mod)" \
--alex_enabled \
--alex_power_weight=1e-1 \
--alex_align_weight=1e-1 \
--alex_fx_smooth_weight=1e-9