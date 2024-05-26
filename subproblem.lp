\ Model Subproblem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 802.9622222222224 y[0,1] + 201.3637777777776 y[0,2]
   + 201.3637777777776 y[1,2] - 802.9622222222224 y[2,1]
   + 201.3637777777776 y[4,2] - 802.9622222222224 y[5,1]
   - 802.9622222222224 y[6,1] + 201.3637777777776 y[6,2] - 41.46125 E[0,1]
   - 55.821 E[0,2] - 52.9895 E[0,3] - 14.562 E[1,2] - 12.944 E[1,3]
   - 14.562 E[2,1] - 4.24725 E[2,3] - 14.562 E[1,5] - 14.562 E[2,4]
   - 52.9895 E[3,6] - 41.46125 E[4,7] - 55.821 E[5,7] - 52.9895 E[6,7]
   - 14.562 E[4,2] - 41.46125 E[4,3] - 14.562 E[5,1] - 55.821 E[5,3]
   - 12.944 E[6,1] - 4.24725 E[6,2] - 14.562 E[4,5] - 12.944 E[4,6]
   - 14.562 E[5,4] - 4.24725 E[5,6] - 12.944 E[6,4] - 4.24725 E[6,5]
   - 273.3333333333334 D[0,1] - 368 D[0,2] - 349.3333333333333 D[0,3]
   - 96 D[1,2] - 85.33333333333334 D[1,3] - 96 D[2,1] - 28 D[2,3]
   - 96 D[1,5] - 96 D[2,4] - 349.3333333333333 D[3,6]
   - 273.3333333333334 D[4,7] - 368 D[5,7] - 349.3333333333333 D[6,7]
   - 96 D[4,2] - 273.3333333333334 D[4,3] - 96 D[5,1] - 368 D[5,3]
   - 85.33333333333334 D[6,1] - 28 D[6,2] - 96 D[4,5]
   - 85.33333333333334 D[4,6] - 96 D[5,4] - 28 D[5,6]
   - 85.33333333333334 D[6,4] - 28 D[6,5] - 29.1 Tdp[1] - 29.1 Tdp[2]
   - 29.1 Tdp[3]
Subject To
 leave_origin_once: y[0,1] + y[0,2] + y[0,3] = 1
 return_destination_once: y[4,7] + y[5,7] + y[6,7] = 1
 flow_conservation_node_1: y[0,1] - y[1,2] - y[1,3] + y[2,1] - y[1,4]
   - y[1,5] + y[5,1] + y[6,1] = 0
 flow_conservation_node_2: y[0,2] + y[1,2] - y[2,1] - y[2,3] - y[2,4]
   - y[2,5] + y[4,2] + y[6,2] = 0
 flow_conservation_node_3: y[0,3] + y[1,3] + y[2,3] - y[3,6] + y[4,3]
   + y[5,3] = 0
 flow_conservation_node_4: y[1,4] + y[2,4] - y[4,7] - y[4,2] - y[4,3]
   - y[4,5] - y[4,6] + y[5,4] + y[6,4] = 0
 flow_conservation_node_5: y[1,5] + y[2,5] - y[5,7] - y[5,1] - y[5,3]
   + y[4,5] - y[5,4] - y[5,6] + y[6,5] = 0
 flow_conservation_node_6: y[3,6] - y[6,7] - y[6,1] - y[6,2] + y[4,6]
   + y[5,6] - y[6,4] - y[6,5] = 0
 pickup_after_delivery_1: y[0,1] + y[2,1] - y[1,4] - y[2,4] + y[5,1]
   + y[6,1] - y[5,4] - y[6,4] = 0
 pickup_after_delivery_2: y[0,2] + y[1,2] - y[1,5] - y[2,5] + y[4,2]
   + y[6,2] - y[4,5] - y[6,5] = 0
 pickup_after_delivery_3: y[0,3] + y[1,3] + y[2,3] - y[3,6] + y[4,3]
   + y[5,3] - y[4,6] - y[5,6] = 0
 visit_turbine_0_once: <= 1
 visit_turbine_1_once: y[0,1] + y[2,1] + y[5,1] + y[6,1] <= 1
 visit_turbine_2_once: y[0,2] + y[1,2] + y[4,2] + y[6,2] <= 1
 visit_turbine_4_once: y[1,4] + y[2,4] + y[5,4] + y[6,4] <= 1
 visit_turbine_5_once: y[1,5] + y[2,5] + y[4,5] + y[6,5] <= 1
 visit_turbine_7_once: y[4,7] + y[5,7] + y[6,7] <= 1
 initial_battery_charge: theta[0] = 60
 max_battery_capacity_0: theta[0] <= 60
 max_battery_capacity_1: theta[1] <= 60
 max_battery_capacity_2: theta[2] <= 60
 max_battery_capacity_3: theta[3] <= 60
 max_battery_capacity_4: theta[4] <= 60
 max_battery_capacity_5: theta[5] <= 60
 max_battery_capacity_6: theta[6] <= 60
 max_battery_capacity_7: theta[7] <= 60
 energy_mix_0_1: - y[0,1] + E[0,1] + D[0,1] = 0
 energy_mix_0_2: - y[0,2] + E[0,2] + D[0,2] = 0
 energy_mix_0_3: - y[0,3] + E[0,3] + D[0,3] = 0
 energy_mix_1_2: - y[1,2] + E[1,2] + D[1,2] = 0
 energy_mix_1_3: - y[1,3] + E[1,3] + D[1,3] = 0
 energy_mix_2_1: - y[2,1] + E[2,1] + D[2,1] = 0
 energy_mix_2_3: - y[2,3] + E[2,3] + D[2,3] = 0
 energy_mix_1_4: - y[1,4] + E[1,4] + D[1,4] = 0
 energy_mix_1_5: - y[1,5] + E[1,5] + D[1,5] = 0
 energy_mix_2_4: - y[2,4] + E[2,4] + D[2,4] = 0
 energy_mix_2_5: - y[2,5] + E[2,5] + D[2,5] = 0
 energy_mix_3_6: - y[3,6] + E[3,6] + D[3,6] = 0
 energy_mix_4_7: - y[4,7] + E[4,7] + D[4,7] = 0
 energy_mix_5_7: - y[5,7] + E[5,7] + D[5,7] = 0
 energy_mix_6_7: - y[6,7] + E[6,7] + D[6,7] = 0
 energy_mix_4_2: - y[4,2] + E[4,2] + D[4,2] = 0
 energy_mix_4_3: - y[4,3] + E[4,3] + D[4,3] = 0
 energy_mix_5_1: - y[5,1] + E[5,1] + D[5,1] = 0
 energy_mix_5_3: - y[5,3] + E[5,3] + D[5,3] = 0
 energy_mix_6_1: - y[6,1] + E[6,1] + D[6,1] = 0
 energy_mix_6_2: - y[6,2] + E[6,2] + D[6,2] = 0
 energy_mix_4_5: - y[4,5] + E[4,5] + D[4,5] = 0
 energy_mix_4_6: - y[4,6] + E[4,6] + D[4,6] = 0
 energy_mix_5_4: - y[5,4] + E[5,4] + D[5,4] = 0
 energy_mix_5_6: - y[5,6] + E[5,6] + D[5,6] = 0
 energy_mix_6_4: - y[6,4] + E[6,4] + D[6,4] = 0
 energy_mix_6_5: - y[6,5] + E[6,5] + D[6,5] = 0
 batteryUpdate0_1: 60 y[0,1] + 20.5 E[0,1] - theta[0] + theta[1] <= 60
 batteryUpdate0_2: 60 y[0,2] + 27.6 E[0,2] - theta[0] + theta[2] <= 60
 batteryUpdate0_3: 60 y[0,3] + 26.2 E[0,3] - theta[0] + theta[3] <= 60
 batteryUpdate1_2: 60 y[1,2] + 7.2 E[1,2] - theta[1] + theta[2]
   + 7.9 Tdp[1] <= 60
 batteryUpdate1_3: 60 y[1,3] + 6.4 E[1,3] - theta[1] + theta[3]
   + 7.9 Tdp[1] <= 60
 batteryUpdate2_1: 60 y[2,1] + 7.2 E[2,1] + theta[1] - theta[2]
   + 7.9 Tdp[2] <= 60
 batteryUpdate2_3: 60 y[2,3] + 2.1 E[2,3] - theta[2] + theta[3]
   + 7.9 Tdp[2] <= 60
 batteryUpdate1_4: 60 y[1,4] - theta[1] + theta[4] + 7.9 Tdp[1] <= 60
 batteryUpdate1_5: 60 y[1,5] + 7.2 E[1,5] - theta[1] + theta[5]
   + 7.9 Tdp[1] <= 60
 batteryUpdate2_4: 60 y[2,4] + 7.2 E[2,4] - theta[2] + theta[4]
   + 7.9 Tdp[2] <= 60
 batteryUpdate2_5: 60 y[2,5] - theta[2] + theta[5] + 7.9 Tdp[2] <= 60
 batteryUpdate4_7: 60 y[4,7] + 20.5 E[4,7] - theta[4] + theta[7] <= 60
 batteryUpdate5_7: 60 y[5,7] + 27.6 E[5,7] - theta[5] + theta[7] <= 60
 batteryUpdate6_7: 60 y[6,7] + 26.2 E[6,7] - theta[6] + theta[7] <= 60
 batteryUpdate4_2: 60 y[4,2] + 7.2 E[4,2] + theta[2] - theta[4] <= 60
 batteryUpdate4_3: 60 y[4,3] + 20.5 E[4,3] + theta[3] - theta[4] <= 60
 batteryUpdate5_1: 60 y[5,1] + 7.2 E[5,1] + theta[1] - theta[5] <= 60
 batteryUpdate5_3: 60 y[5,3] + 27.6 E[5,3] + theta[3] - theta[5] <= 60
 batteryUpdate6_1: 60 y[6,1] + 6.4 E[6,1] + theta[1] - theta[6] <= 60
 batteryUpdate6_2: 60 y[6,2] + 2.1 E[6,2] + theta[2] - theta[6] <= 60
 batteryUpdate4_5: 60 y[4,5] + 7.2 E[4,5] - theta[4] + theta[5] <= 60
 batteryUpdate4_6: 60 y[4,6] + 6.4 E[4,6] - theta[4] + theta[6] <= 60
 batteryUpdate5_4: 60 y[5,4] + 7.2 E[5,4] + theta[4] - theta[5] <= 60
 batteryUpdate5_6: 60 y[5,6] + 2.1 E[5,6] - theta[5] + theta[6] <= 60
 batteryUpdate6_4: 60 y[6,4] + 6.4 E[6,4] + theta[4] - theta[6] <= 60
 batteryUpdate6_5: 60 y[6,5] + 2.1 E[6,5] + theta[5] - theta[6] <= 60
 charging_at_node_3: theta[3] - theta[6] - 52 q[3] + 52 q[6] >= 26
 leave_origin_on_eletric: E[0,1] + E[0,2] + E[0,3] >= 0.05
 enter_base_on_eletric: E[4,7] + E[5,7] + E[6,7] >= 0.05
 return_to_destination: q[7] <= 12
 time_update_0_1: 12.75925925925926 y[0,1] + q[0] - q[1] <= 12
 time_update_0_2: 13.02222222222222 y[0,2] + q[0] - q[2] <= 12
 time_update_0_3: 12.97037037037037 y[0,3] + q[0] - q[3] <= 12
 time_update_1_2: 12.26666666666667 y[1,2] + q[1] - q[2] <= 12
 time_update_1_3: 12.23703703703704 y[1,3] + q[1] - q[3] <= 12
 time_update_2_1: 12.26666666666667 y[2,1] - q[1] + q[2] <= 12
 time_update_2_3: 12.07777777777778 y[2,3] + q[2] - q[3] <= 12
 time_update_1_4: 12 y[1,4] + q[1] - q[4] <= 12
 time_update_1_5: 12.26666666666667 y[1,5] + q[1] - q[5] <= 12
 time_update_2_4: 12.26666666666667 y[2,4] + q[2] - q[4] <= 12
 time_update_2_5: 12 y[2,5] + q[2] - q[5] <= 12
 time_update_3_6: 12.97037037037037 y[3,6] + q[3] - q[6] <= 12
 time_update_4_7: 12.75925925925926 y[4,7] + q[4] - q[7] <= 12
 time_update_5_7: 13.02222222222222 y[5,7] + q[5] - q[7] <= 12
 time_update_6_7: 12.97037037037037 y[6,7] + q[6] - q[7] <= 12
 time_update_4_2: 12.26666666666667 y[4,2] - q[2] + q[4] <= 12
 time_update_4_3: 12.75925925925926 y[4,3] - q[3] + q[4] <= 12
 time_update_5_1: 12.26666666666667 y[5,1] - q[1] + q[5] <= 12
 time_update_5_3: 13.02222222222222 y[5,3] - q[3] + q[5] <= 12
 time_update_6_1: 12.23703703703704 y[6,1] - q[1] + q[6] <= 12
 time_update_6_2: 12.07777777777778 y[6,2] - q[2] + q[6] <= 12
 time_update_4_5: 12.26666666666667 y[4,5] + q[4] - q[5] <= 12
 time_update_4_6: 12.23703703703704 y[4,6] + q[4] - q[6] <= 12
 time_update_5_4: 12.26666666666667 y[5,4] - q[4] + q[5] <= 12
 time_update_5_6: 12.07777777777778 y[5,6] + q[5] - q[6] <= 12
 time_update_6_4: 12.23703703703704 y[6,4] - q[4] + q[6] <= 12
 time_update_6_5: 12.07777777777778 y[6,5] - q[5] + q[6] <= 12
 adequate_time_for_tasks_1: q[1] - q[4] <= -5.71
 adequate_time_for_tasks_2: q[2] - q[5] <= -2.35
 R112: 12 y[1,4] - Tdp[1] - q[1] + q[4] <= 12
 R113: 12 y[2,5] - Tdp[2] - q[2] + q[5] <= 12
 R114: 12 y[3,6] - Tdp[3] - q[3] + q[6] <= 12
 vessel_tech_capacity: z[1,0] + z[2,0] <= 24
 tech_availibility: z[1,0] <= 24
 tech_availibility: z[2,0] <= 24
 tech_availability_meet_demand_1_0_1: 24 y[0,1] + z[1,0] - z[1,1] <= 29
 tech_availability_no_overestimate_1_0_1: - 24 y[0,1] + z[1,0] - z[1,1]
   >= -19
 tech_availability_meet_demand_1_0_2: 24 y[0,2] + z[1,0] - z[1,2] <= 26
 tech_availability_no_overestimate_1_0_2: - 24 y[0,2] + z[1,0] - z[1,2]
   >= -22
 tech_availability_meet_demand_1_0_3: 24 y[0,3] + z[1,0] - z[1,3] <= 24
 tech_availability_no_overestimate_1_0_3: - 24 y[0,3] + z[1,0] - z[1,3]
   >= -24
 tech_availability_meet_demand_1_1_2: 24 y[1,2] + z[1,1] - z[1,2] <= 26
 tech_availability_no_overestimate_1_1_2: - 24 y[1,2] + z[1,1] - z[1,2]
   >= -22
 tech_availability_meet_demand_1_1_3: 24 y[1,3] + z[1,1] - z[1,3] <= 24
 tech_availability_no_overestimate_1_1_3: - 24 y[1,3] + z[1,1] - z[1,3]
   >= -24
 tech_availability_meet_demand_1_2_1: 24 y[2,1] - z[1,1] + z[1,2] <= 29
 tech_availability_no_overestimate_1_2_1: - 24 y[2,1] - z[1,1] + z[1,2]
   >= -19
 tech_availability_meet_demand_1_2_3: 24 y[2,3] + z[1,2] - z[1,3] <= 24
 tech_availability_no_overestimate_1_2_3: - 24 y[2,3] + z[1,2] - z[1,3]
   >= -24
 tech_availability_meet_demand_1_1_4: 24 y[1,4] + z[1,1] - z[1,4] <= 19
 tech_availability_no_overestimate_1_1_4: - 24 y[1,4] + z[1,1] - z[1,4]
   >= -29
 tech_availability_meet_demand_1_1_5: 24 y[1,5] + z[1,1] - z[1,5] <= 22
 tech_availability_no_overestimate_1_1_5: - 24 y[1,5] + z[1,1] - z[1,5]
   >= -26
 tech_availability_meet_demand_1_2_4: 24 y[2,4] + z[1,2] - z[1,4] <= 19
 tech_availability_no_overestimate_1_2_4: - 24 y[2,4] + z[1,2] - z[1,4]
   >= -29
 tech_availability_meet_demand_1_2_5: 24 y[2,5] + z[1,2] - z[1,5] <= 22
 tech_availability_no_overestimate_1_2_5: - 24 y[2,5] + z[1,2] - z[1,5]
   >= -26
 tech_availability_meet_demand_1_3_6: 24 y[3,6] + z[1,3] - z[1,6] <= 24
 tech_availability_no_overestimate_1_3_6: - 24 y[3,6] + z[1,3] - z[1,6]
   >= -24
 tech_availability_meet_demand_1_4_7: 24 y[4,7] + z[1,4] - z[1,7] <= 24
 tech_availability_no_overestimate_1_4_7: - 24 y[4,7] + z[1,4] - z[1,7]
   >= -24
 tech_availability_meet_demand_1_5_7: 24 y[5,7] + z[1,5] - z[1,7] <= 24
 tech_availability_no_overestimate_1_5_7: - 24 y[5,7] + z[1,5] - z[1,7]
   >= -24
 tech_availability_meet_demand_1_6_7: 24 y[6,7] + z[1,6] - z[1,7] <= 24
 tech_availability_no_overestimate_1_6_7: - 24 y[6,7] + z[1,6] - z[1,7]
   >= -24
 tech_availability_meet_demand_1_4_2: 24 y[4,2] - z[1,2] + z[1,4] <= 26
 tech_availability_no_overestimate_1_4_2: - 24 y[4,2] - z[1,2] + z[1,4]
   >= -22
 tech_availability_meet_demand_1_4_3: 24 y[4,3] - z[1,3] + z[1,4] <= 24
 tech_availability_no_overestimate_1_4_3: - 24 y[4,3] - z[1,3] + z[1,4]
   >= -24
 tech_availability_meet_demand_1_5_1: 24 y[5,1] - z[1,1] + z[1,5] <= 29
 tech_availability_no_overestimate_1_5_1: - 24 y[5,1] - z[1,1] + z[1,5]
   >= -19
 tech_availability_meet_demand_1_5_3: 24 y[5,3] - z[1,3] + z[1,5] <= 24
 tech_availability_no_overestimate_1_5_3: - 24 y[5,3] - z[1,3] + z[1,5]
   >= -24
 tech_availability_meet_demand_1_6_1: 24 y[6,1] - z[1,1] + z[1,6] <= 29
 tech_availability_no_overestimate_1_6_1: - 24 y[6,1] - z[1,1] + z[1,6]
   >= -19
 tech_availability_meet_demand_1_6_2: 24 y[6,2] - z[1,2] + z[1,6] <= 26
 tech_availability_no_overestimate_1_6_2: - 24 y[6,2] - z[1,2] + z[1,6]
   >= -22
 tech_availability_meet_demand_1_4_5: 24 y[4,5] + z[1,4] - z[1,5] <= 22
 tech_availability_no_overestimate_1_4_5: - 24 y[4,5] + z[1,4] - z[1,5]
   >= -26
 tech_availability_meet_demand_1_4_6: 24 y[4,6] + z[1,4] - z[1,6] <= 24
 tech_availability_no_overestimate_1_4_6: - 24 y[4,6] + z[1,4] - z[1,6]
   >= -24
 tech_availability_meet_demand_1_5_4: 24 y[5,4] - z[1,4] + z[1,5] <= 19
 tech_availability_no_overestimate_1_5_4: - 24 y[5,4] - z[1,4] + z[1,5]
   >= -29
 tech_availability_meet_demand_1_5_6: 24 y[5,6] + z[1,5] - z[1,6] <= 24
 tech_availability_no_overestimate_1_5_6: - 24 y[5,6] + z[1,5] - z[1,6]
   >= -24
 tech_availability_meet_demand_1_6_4: 24 y[6,4] - z[1,4] + z[1,6] <= 19
 tech_availability_no_overestimate_1_6_4: - 24 y[6,4] - z[1,4] + z[1,6]
   >= -29
 tech_availability_meet_demand_1_6_5: 24 y[6,5] - z[1,5] + z[1,6] <= 22
 tech_availability_no_overestimate_1_6_5: - 24 y[6,5] - z[1,5] + z[1,6]
   >= -26
 tech_availability_meet_demand_2_0_1: 24 y[0,1] + z[2,0] - z[2,1] <= 27
 tech_availability_no_overestimate_2_0_1: - 24 y[0,1] + z[2,0] - z[2,1]
   >= -21
 tech_availability_meet_demand_2_0_2: 24 y[0,2] + z[2,0] - z[2,2] <= 27
 tech_availability_no_overestimate_2_0_2: - 24 y[0,2] + z[2,0] - z[2,2]
   >= -21
 tech_availability_meet_demand_2_0_3: 24 y[0,3] + z[2,0] - z[2,3] <= 24
 tech_availability_no_overestimate_2_0_3: - 24 y[0,3] + z[2,0] - z[2,3]
   >= -24
 tech_availability_meet_demand_2_1_2: 24 y[1,2] + z[2,1] - z[2,2] <= 27
 tech_availability_no_overestimate_2_1_2: - 24 y[1,2] + z[2,1] - z[2,2]
   >= -21
 tech_availability_meet_demand_2_1_3: 24 y[1,3] + z[2,1] - z[2,3] <= 24
 tech_availability_no_overestimate_2_1_3: - 24 y[1,3] + z[2,1] - z[2,3]
   >= -24
 tech_availability_meet_demand_2_2_1: 24 y[2,1] - z[2,1] + z[2,2] <= 27
 tech_availability_no_overestimate_2_2_1: - 24 y[2,1] - z[2,1] + z[2,2]
   >= -21
 tech_availability_meet_demand_2_2_3: 24 y[2,3] + z[2,2] - z[2,3] <= 24
 tech_availability_no_overestimate_2_2_3: - 24 y[2,3] + z[2,2] - z[2,3]
   >= -24
 tech_availability_meet_demand_2_1_4: 24 y[1,4] + z[2,1] - z[2,4] <= 21
 tech_availability_no_overestimate_2_1_4: - 24 y[1,4] + z[2,1] - z[2,4]
   >= -27
 tech_availability_meet_demand_2_1_5: 24 y[1,5] + z[2,1] - z[2,5] <= 21
 tech_availability_no_overestimate_2_1_5: - 24 y[1,5] + z[2,1] - z[2,5]
   >= -27
 tech_availability_meet_demand_2_2_4: 24 y[2,4] + z[2,2] - z[2,4] <= 21
 tech_availability_no_overestimate_2_2_4: - 24 y[2,4] + z[2,2] - z[2,4]
   >= -27
 tech_availability_meet_demand_2_2_5: 24 y[2,5] + z[2,2] - z[2,5] <= 21
 tech_availability_no_overestimate_2_2_5: - 24 y[2,5] + z[2,2] - z[2,5]
   >= -27
 tech_availability_meet_demand_2_3_6: 24 y[3,6] + z[2,3] - z[2,6] <= 24
 tech_availability_no_overestimate_2_3_6: - 24 y[3,6] + z[2,3] - z[2,6]
   >= -24
 tech_availability_meet_demand_2_4_7: 24 y[4,7] + z[2,4] - z[2,7] <= 24
 tech_availability_no_overestimate_2_4_7: - 24 y[4,7] + z[2,4] - z[2,7]
   >= -24
 tech_availability_meet_demand_2_5_7: 24 y[5,7] + z[2,5] - z[2,7] <= 24
 tech_availability_no_overestimate_2_5_7: - 24 y[5,7] + z[2,5] - z[2,7]
   >= -24
 tech_availability_meet_demand_2_6_7: 24 y[6,7] + z[2,6] - z[2,7] <= 24
 tech_availability_no_overestimate_2_6_7: - 24 y[6,7] + z[2,6] - z[2,7]
   >= -24
 tech_availability_meet_demand_2_4_2: 24 y[4,2] - z[2,2] + z[2,4] <= 27
 tech_availability_no_overestimate_2_4_2: - 24 y[4,2] - z[2,2] + z[2,4]
   >= -21
 tech_availability_meet_demand_2_4_3: 24 y[4,3] - z[2,3] + z[2,4] <= 24
 tech_availability_no_overestimate_2_4_3: - 24 y[4,3] - z[2,3] + z[2,4]
   >= -24
 tech_availability_meet_demand_2_5_1: 24 y[5,1] - z[2,1] + z[2,5] <= 27
 tech_availability_no_overestimate_2_5_1: - 24 y[5,1] - z[2,1] + z[2,5]
   >= -21
 tech_availability_meet_demand_2_5_3: 24 y[5,3] - z[2,3] + z[2,5] <= 24
 tech_availability_no_overestimate_2_5_3: - 24 y[5,3] - z[2,3] + z[2,5]
   >= -24
 tech_availability_meet_demand_2_6_1: 24 y[6,1] - z[2,1] + z[2,6] <= 27
 tech_availability_no_overestimate_2_6_1: - 24 y[6,1] - z[2,1] + z[2,6]
   >= -21
 tech_availability_meet_demand_2_6_2: 24 y[6,2] - z[2,2] + z[2,6] <= 27
 tech_availability_no_overestimate_2_6_2: - 24 y[6,2] - z[2,2] + z[2,6]
   >= -21
 tech_availability_meet_demand_2_4_5: 24 y[4,5] + z[2,4] - z[2,5] <= 21
 tech_availability_no_overestimate_2_4_5: - 24 y[4,5] + z[2,4] - z[2,5]
   >= -27
 tech_availability_meet_demand_2_4_6: 24 y[4,6] + z[2,4] - z[2,6] <= 24
 tech_availability_no_overestimate_2_4_6: - 24 y[4,6] + z[2,4] - z[2,6]
   >= -24
 tech_availability_meet_demand_2_5_4: 24 y[5,4] - z[2,4] + z[2,5] <= 21
 tech_availability_no_overestimate_2_5_4: - 24 y[5,4] - z[2,4] + z[2,5]
   >= -27
 tech_availability_meet_demand_2_5_6: 24 y[5,6] + z[2,5] - z[2,6] <= 24
 tech_availability_no_overestimate_2_5_6: - 24 y[5,6] + z[2,5] - z[2,6]
   >= -24
 tech_availability_meet_demand_2_6_4: 24 y[6,4] - z[2,4] + z[2,6] <= 21
 tech_availability_no_overestimate_2_6_4: - 24 y[6,4] - z[2,4] + z[2,6]
   >= -27
 tech_availability_meet_demand_2_6_5: 24 y[6,5] - z[2,5] + z[2,6] <= 21
 tech_availability_no_overestimate_2_6_5: - 24 y[6,5] - z[2,5] + z[2,6]
   >= -27
Bounds
 E[0,1] <= 1
 E[0,2] <= 1
 E[0,3] <= 1
 E[1,2] <= 1
 E[1,3] <= 1
 E[2,1] <= 1
 E[2,3] <= 1
 E[1,4] <= 1
 E[1,5] <= 1
 E[2,4] <= 1
 E[2,5] <= 1
 E[3,6] <= 1
 E[4,7] <= 1
 E[5,7] <= 1
 E[6,7] <= 1
 E[4,2] <= 1
 E[4,3] <= 1
 E[5,1] <= 1
 E[5,3] <= 1
 E[6,1] <= 1
 E[6,2] <= 1
 E[4,5] <= 1
 E[4,6] <= 1
 E[5,4] <= 1
 E[5,6] <= 1
 E[6,4] <= 1
 E[6,5] <= 1
 D[0,1] <= 1
 D[0,2] <= 1
 D[0,3] <= 1
 D[1,2] <= 1
 D[1,3] <= 1
 D[2,1] <= 1
 D[2,3] <= 1
 D[1,4] <= 1
 D[1,5] <= 1
 D[2,4] <= 1
 D[2,5] <= 1
 D[3,6] <= 1
 D[4,7] <= 1
 D[5,7] <= 1
 D[6,7] <= 1
 D[4,2] <= 1
 D[4,3] <= 1
 D[5,1] <= 1
 D[5,3] <= 1
 D[6,1] <= 1
 D[6,2] <= 1
 D[4,5] <= 1
 D[4,6] <= 1
 D[5,4] <= 1
 D[5,6] <= 1
 D[6,4] <= 1
 D[6,5] <= 1
 theta[0] <= 60
 theta[1] <= 60
 theta[2] <= 60
 theta[3] <= 60
 theta[4] <= 60
 theta[5] <= 60
 theta[6] <= 60
 theta[7] <= 60
Binaries
 y[0,1] y[0,2] y[0,3] y[1,2] y[1,3] y[2,1] y[2,3] y[1,4] y[1,5] y[2,4]
 y[2,5] y[3,6] y[4,7] y[5,7] y[6,7] y[4,2] y[4,3] y[5,1] y[5,3] y[6,1]
 y[6,2] y[4,5] y[4,6] y[5,4] y[5,6] y[6,4] y[6,5]
Generals
 z[1,0] z[1,1] z[1,2] z[1,3] z[1,4] z[1,5] z[1,6] z[1,7] z[2,0] z[2,1]
 z[2,2] z[2,3] z[2,4] z[2,5] z[2,6] z[2,7]
End
