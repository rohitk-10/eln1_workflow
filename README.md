# Workflow for ELAIS-N1 and Deep Fields sources
Initial workflow/flowchart developed for the ELAIS-N1 and the LOFAR deep fields to decide which sources can be cross-matched via statistical sources and those to be sent to LGZ

## Description
Script uses output of the Likelihood Ratio analysis and categories how to classify the source (LR-ID or visual) based on various properties such as LR value, size, clustering and source morphology.

# Usage
Use the eln1_workflow.py code
Edit the paths to point to the calibrated q(m) and thresholds and the paths to the LR matches catalogue.

* Can be run in a few different ways:
1. Selecting sources most suitable for LR:
	- Set "write_out" = True to write output
	- Set workflow_iter = False - You don't wait to use output from a previous iteration of the flowchart. As you are running the flowchart for the first time to find the most suitable sources for LR calibration, this is the appropriate choice.
	- When prompted by code, press "y" to find sources most suitable for LR

	- Output will be in workflow_iter_N/ directory where N corresponds to the Nth time that the output is written 

2. Selecting sources that need to be send to visual analysis
	- Set "write_out" = True to write output
	- Set workflow_iter = True - as you are now using LR output after step 1, i.e. after you have calibrated on sources most suitable for LR
	- When prompted by code, press "n" to choose to not calibrate LR
	- This created a FLAG_WORKFLOW column with flags for the different endpoints for each radio source
		-- FLAG_WORKFLOW = 1; Accept LR-ID
		-- FLAG_WORKFLOW = 2; Send to LGZ
		-- FLAG_WORKFLOW = 3; Pre-filter1 (workflow_3)
		-- FLAG_WORKFLOW = 4; Pre-filter2 (workflow_4)
		-- FLAG_WORKFLOW = 5; Deblending
	- Output will be in iterated_endpoints/ directory (BEWARE - this is overwritten if code is run again in this mode!)
