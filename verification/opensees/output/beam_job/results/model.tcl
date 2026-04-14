wipe
model BasicBuilder -ndm 2 -ndf 3

node 1 0 0.0
node 2 0.510833333333 0.0
node 3 1.02166666667 0.0
node 4 1.5325 0.0
node 5 2.04333333333 0.0
node 6 2.55416666667 0.0
node 7 3.065 0.0
node 8 3.57583333333 0.0
node 9 4.08666666667 0.0
node 10 4.5975 0.0
node 11 5.10833333333 0.0
node 12 5.61916666667 0.0
node 13 6.13 0.0
fix 1 1 1 1

geomTransf Linear 1
element elasticBeamColumn 1 1 2 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 2 2 3 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 3 3 4 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 4 4 5 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 5 5 6 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 6 6 7 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 7 7 8 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 8 8 9 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 9 9 10 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 10 10 11 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 11 11 12 0.000764 25000000000 8.489e-08 1
element elasticBeamColumn 12 12 13 0.000764 25000000000 8.489e-08 1

timeSeries Linear 1
pattern Plain 1 1 {
    load 13 0 -1350 0
}

recorder Node -file "node_displacements.out" -nodeRange 1 13 -dof 1 2 3 disp
recorder Element -file "element_forces.out" -eleRange 1 12 localForce

constraints Plain
numberer RCM
system BandGeneral
test NormDispIncr 1.0e-10 50 0
algorithm Newton
integrator LoadControl 1.0
analysis Static

set analysisCode [analyze 1]
if {$analysisCode != 0} {
    puts "Analysis failed with code $analysisCode"
    exit $analysisCode
}

reactions
set supportReactions [open "support_reactions.out" "w"]
puts $supportReactions "1 [nodeReaction 1 1] [nodeReaction 1 2] [nodeReaction 1 3]"
close $supportReactions

set responseDispX [nodeDisp 13 1]
set responseDispY [nodeDisp 13 2]
set responseRotZ [nodeDisp 13 3]
set summary [open "summary.txt" "w"]
puts $summary "case_type cantilever_tip_load"
puts $summary "load_node 13"
puts $summary "load_disp_x $responseDispX"
puts $summary "load_disp_y $responseDispY"
puts $summary "load_rot_z $responseRotZ"
close $summary

wipe
