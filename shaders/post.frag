/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;
layout(set = 0, binding = 1) uniform sampler2D aoTxt;


layout(push_constant) uniform shaderInformation
{
  float aspectRatio;
}
pushc;

void main()
{
  vec2  uv    = outUV;
  vec3 gamma = vec3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2);
  vec4 color = texture(noisyTxt, uv);
  vec3 GI = texture(aoTxt, uv).xyz;
  vec3 shaded_color = GI;

  shaded_color = shaded_color / (shaded_color + vec3(1));

  vec3 gamma_corrected = pow(shaded_color, gamma);

  fragColor =  vec4(gamma_corrected.xyz, color.a);
}
