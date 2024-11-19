#Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: Apache-2.0

import os

import sys
import traceback
import openvino as ov

import io

import logging

import time

logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)

def compile_and_export_model(core, model_path, output_path, device='NPU', config=None):
    try:
        # Compile the model for the specified device
        print("Step 3: Starting Model Compile for NPU..")
        t0 = time.time()
        model = core.compile_model(model_path, device, config=config)
        t1 = time.time()
        print("Model compile took", t1 - t0, "s")

        # Export the compiled model to a binary blob
        with io.BytesIO() as model_blob:
            model.export_model(model_blob)

            # Write the binary blob to the output path
            temp_output_path = str(output_path) + ".tmp"
            with open(temp_output_path, 'wb') as f:
                f.write(model_blob.getvalue())

            # Remove the existing file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)

            # Rename the temporary file to the final output path
            os.rename(temp_output_path, output_path)

        logging.info(f"Model compiled and exported successfully to {output_path}")

    except Exception as e:
        logging.error(f"Failed to compile and export model: {str(e)}")
        tb_str = traceback.format_exc()
        raise RuntimeError(f"Model compilation or export failed for {model_path} on device {device}.\nDetails: {tb_str}")
    


