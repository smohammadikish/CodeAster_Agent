from __future__ import annotations

from pathlib import Path

from ver_opensees.models import BeamInput


# Write the generated Tcl model to a file.
def write_model_tcl(beam_input: BeamInput, target: Path) -> None:
    target.write_text(_build_tcl(beam_input), encoding="utf-8")


# Build the OpenSees Tcl script for a 2D cantilever beam.
def _build_tcl(beam_input: BeamInput) -> str:
    # Calculate the element length and the main node ids.
    dx = beam_input.length / beam_input.num_elements
    last_node = beam_input.num_nodes

    # Create all beam nodes along the x direction.
    node_lines = [
        f"node {node_id} {(node_id - 1) * dx:.12g} 0.0"
        for node_id in range(1, beam_input.num_nodes + 1)
    ]

    # Create elastic beam-column elements between the nodes.
    element_lines = [
        (
            "element elasticBeamColumn "
            f"{element_id} {element_id} {element_id + 1} "
            f"{beam_input.section.area:.12g} "
            f"{beam_input.material.elastic_modulus:.12g} "
            f"{beam_input.section.inertia_z:.12g} 1"
        )
        for element_id in range(1, beam_input.num_elements + 1)
    ]

    # Create the support commands based on the selected beam type.
    support_lines = _build_support_lines(beam_input, last_node)
    support_nodes = _support_node_ids(beam_input, last_node)

    # Return the full Tcl model as one text block.
    return "\n".join(
        [
            # Start a clean OpenSees model and define a 2D frame.
            "wipe",
            "model BasicBuilder -ndm 2 -ndf 3",
            "",
            # Add all nodes and apply the selected support condition.
            *node_lines,
            *support_lines,
            "",
            # Define the geometric transformation and all beam elements.
            "geomTransf Linear 1",
            *element_lines,
            "",
            # Define the load pattern and apply the nodal load.
            "timeSeries Linear 1",
            "pattern Plain 1 1 {",
            (
                f"    load {beam_input.load.node} "
                f"{beam_input.load.fx:.12g} "
                f"{beam_input.load.fy:.12g} "
                f"{beam_input.load.mz:.12g}"
            ),
            "}",
            "",
            # Save nodal displacements and element forces to output files.
            'recorder Node -file "node_displacements.out" -nodeRange 1 '
            f"{beam_input.num_nodes} -dof 1 2 3 disp",
            'recorder Element -file "element_forces.out" -eleRange 1 '
            f"{beam_input.num_elements} localForce",
            "",
            # Set the standard options for a simple static analysis.
            "constraints Plain",
            "numberer RCM",
            "system BandGeneral",
            "test NormDispIncr 1.0e-10 50 0",
            "algorithm Newton",
            "integrator LoadControl 1.0",
            "analysis Static",
            "",
            # Run the analysis and stop if OpenSees reports an error.
            "set analysisCode [analyze 1]",
            "if {$analysisCode != 0} {",
            '    puts "Analysis failed with code $analysisCode"',
            "    exit $analysisCode",
            "}",
            "",
            # Ask OpenSees to assemble reactions and save them for support nodes.
            "reactions",
            'set supportReactions [open "support_reactions.out" "w"]',
            *[
                (
                    f'puts $supportReactions "{node_id} '
                    f'[nodeReaction {node_id} 1] '
                    f'[nodeReaction {node_id} 2] '
                    f'[nodeReaction {node_id} 3]"'
                )
                for node_id in support_nodes
            ],
            "close $supportReactions",
            "",
            # Read the selected node response and write a short summary file.
            f"set responseDispX [nodeDisp {beam_input.load.node} 1]",
            f"set responseDispY [nodeDisp {beam_input.load.node} 2]",
            f"set responseRotZ [nodeDisp {beam_input.load.node} 3]",
            'set summary [open "summary.txt" "w"]',
            f'puts $summary "case_type {beam_input.case_type}"',
            f'puts $summary "load_node {beam_input.load.node}"',
            'puts $summary "load_disp_x $responseDispX"',
            'puts $summary "load_disp_y $responseDispY"',
            'puts $summary "load_rot_z $responseRotZ"',
            "close $summary",
            "",
            # Clear the model at the end of the run.
            "wipe",
            "",
        ]
    )


# Build the OpenSees support commands for the selected beam type.
def _build_support_lines(beam_input: BeamInput, last_node: int) -> list[str]:
    if beam_input.support_type == "cantilever":
        return ["fix 1 1 1 1"]

    return [
        "fix 1 1 1 0",
        f"fix {last_node} 0 1 0",
    ]


def _support_node_ids(beam_input: BeamInput, last_node: int) -> list[int]:
    if beam_input.support_type == "cantilever":
        return [1]

    return [1, last_node]
