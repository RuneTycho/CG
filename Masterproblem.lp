\ Model Master Problem IP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  5676.316666666667 route[1] + 6612.316666666667 route[2]
   + 5676.316666666667 route[3] + 6612.316666666667 route[4]
   + 1811.636222222222 route[5] + 3606.279379166665 route[6]
   + 1811.636222222222 route[7] + 3606.279379166665 route[8]
   + 3723.962222222222 route[9] + 3740.636222222222 route[10]
   + 3723.962222222222 route[11] + 3740.636222222222 route[12]
Subject To
 tech_type_1_time_window_1_availability: 7 route[1] + 7 route[3]
   + 2 route[5] + 2 route[7] + 5 route[9] + 5 route[11] <= 24
 tech_type_2_time_window_1_availability: 6 route[1] + 6 route[3]
   + 3 route[5] + 3 route[7] + 3 route[9] + 3 route[11] <= 24
 tech_type_1_time_window_2_availability: 7 route[2] + 7 route[4]
   + 2 route[6] + 2 route[8] + 2 route[10] + 2 route[12] <= 24
 tech_type_2_time_window_2_availability: 6 route[2] + 6 route[4]
   + 3 route[6] + 3 route[8] + 3 route[10] + 3 route[12] <= 24
 job_1_once: route[1] + route[2] + route[3] + route[4] + route[9]
   + route[11] <= 1
 job_2_once: route[1] + route[2] + route[3] + route[4] + route[5]
   + route[6] + route[7] + route[8] + route[10] + route[12] <= 1
 one_route_per_vessel_Vessel_1_tw_1: route[1] + route[5] + route[9] <= 1
 one_route_per_vessel_Vessel_1_tw_2: route[2] + route[6] + route[10] <= 1
 one_route_per_vessel_Vessel_2_tw_1: route[3] + route[7] + route[11] <= 1
 one_route_per_vessel_Vessel_2_tw_2: route[4] + route[8] + route[12] <= 1
Bounds
Binaries
 route[1] route[2] route[3] route[4] route[5] route[6] route[7] route[8]
 route[9] route[10] route[11] route[12]
End
