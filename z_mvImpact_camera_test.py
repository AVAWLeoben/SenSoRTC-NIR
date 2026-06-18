#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:03:42 2026

@author: admin
"""

from mvimpact_nir_camera import MvImpactNIRCamera

cam = MvImpactNIRCamera(
    settings_path="mvimpact_nir_camera_settings.yaml",
    classifier_path="PE_PP_PET_SNN.joblib",  # or your .joblib classifier
).connect()

for _ in range(2):
    ret, frame = cam.read()
    print("ret:", ret)
    print("frame:", None if frame is None else frame.shape)
    print("line:", cam.last_classified_line.shape, cam.last_classified_line[:20])

cam.release()