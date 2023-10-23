Shader "Custom/CloudsFull" {
  Properties {
    _MainTex ("Texture", 2D) = "white" {}

    _PlanetRadius ("Planet Radius", Float) = 6000000
    _CloudLayerStart ("Cloud Layer Start", Float) = 400
    _CloudLayerEnd ("Cloud Layer End", Float) = 1000

    [HDR] _AmbientBottomColor ("Bottom Ambient Color", Color) = (0.5, 0.67, 0.82, 1)
    [HDR] _AmbientTopColor ("Top Ambient Color", Color) = (1, 1, 1, 1)
    _FogColor ("Fog Color", Color) = (0.3, 0.4, 0.45, 1)

    _CloudShapeNoiseScale ("Shape Noise Scale", Float) = 1

    _WeatherMapScrollingSpeed ("Weather Map Scrolling Speed", Float) = 1
    _LowNoiseScrollingSpeed ("Shape Noise Scrolling Speed", Float) = 1
    _HighNoiseScrollingSpeed ("Detail Noise Scrolling Speed", Float) = 1

    _WeatherMap ("Weather Map", 2D) = "white" {}
    _WeatherMapScale ("Weather Map Scale", Float) = 20000
    _WeatherMapOffset ("Weather Map Offset", Vector) = (0, 0, 0, 0)

    _CloudCoverageMultiplier ("Cloud Coverage Multiplier", Float) = 1
    _CloudCoverageMinimum ("Cloud Coverage Minimum", Float) = 0
    _CloudTypeMultiplier ("Cloud Type Multiplier", Float) = 1
    _CloudDensityCoverageMultiplier ("Cloud Density Coverage Multiplier", Float) = 1

    _DensityErosionTex ("Density Erosion Texture", 2D) = "white" {}

    _DetailStrength ("Detail Strength", Float) = 1
    _DetailNoiseScale ("Detail Noise Scale", Float) = 1

    _FadeMinDistance ("Cloud Min Fade Distance", Float) = 10000
    _FadeMaxDistance ("Cloud Max Fade Distance", Float) = 20000
    _FadeHorizonAngle ("Cloud Below Horizon Fade Angle", Float) = 0.01

    _CloudNoiseLowFreq ("Low Frequency Cloud Noise", 3D) = "white" {}

    _CloudNoiseHighFreq ("High Frequency Cloud Noise", 3D) = "white" {}

    _CloudChunkiness ("Cloud Chunkiness", Float) = 4
    _SilverLiningPower ("Silver Lining Power", Float) = 1.1
    _PowderSugarStrength ("Powder Sugar Effect Strength", Float) = 1
    _BeerLambertStrength ("Beer Lambert Strength", Float) = 1

    _CurlTex ("Curl Texture", 2D) = "white" {}
    _CurlStrength ("Curl Strength", Float) = 1
    _CurlScale ("Curl Scale", Float) = 1

    _BlueNoise("Blue Noise", 2D) = "white" {}
    _BlueNoiseStrength("Blue Noise Strength", Float) = 1

    _LightAbsorbtion ("Light Absorbtion", Float) = 6
  }

  SubShader {
    Tags { "RenderType" = "Transparent"
           "RenderPipeline" = "UniversalPipeline"
           "Queue" = "Overlay"
           "UniversalMaterialType" = "Lit" }

    HLSLINCLUDE
    #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
    ENDHLSL

    Pass {
      Name "ForwardLit"
      Tags { "LightMode" = "UniversalForward" }

      HLSLPROGRAM
      #pragma prefer_hlslcc gles
      #pragma exclude_renderers d3d11_9x
      #pragma target 3.0

      #pragma vertex vert
      #pragma fragment frag

      CBUFFER_START(UnityPerMaterial)
      float _GlobalCoverage;
      float _GlobalDensity;

      float _LightAbsorbtion;

      float _PlanetRadius;
      float _CloudLayerStart;
      float _CloudLayerEnd;

      float _CloudShapeNoiseScale;
      float _CloudCoverageTest;
      float _CloudHeightTest;
      float2 _A;

      float _CloudCoverageMinimum;
      float _CloudCoverageMultiplier;
      float _CloudTypeMultiplier;
      float _CloudDensityCoverageMultiplier;

      float3 _AmbientBottomColor;
      float3 _AmbientTopColor;
      float3 _FogColor;

      float _DetailStrength;
      float _DetailNoiseScale;

      float _CurlStrength;
      float _CurlScale;

      float _FadeMinDistance;
      float _FadeMaxDistance;
      float _FadeHorizonAngle;

      float _CloudChunkiness;
      float _SilverLiningPower;
      float _PowderSugarStrength;
      float _BeerLambertStrength;

      float _WeatherMapScrollingSpeed;
      float _LowNoiseScrollingSpeed;
      float _HighNoiseScrollingSpeed;

      float _BlueNoiseStrength;
      
      TEXTURE2D(_MainTex);
      SAMPLER(sampler_MainTex);
      float4 _MainTex_TexelSize;

      TEXTURE2D(_WeatherMap);
      float4 _WeatherMap_ST;
      SAMPLER(sampler_WeatherMap);
      float _WeatherMapScale;

      TEXTURE3D(_CloudNoiseLowFreq);
      float4 _CloudNoiseLowFreq_ST;
      SAMPLER(sampler_CloudNoiseLowFreq);

      TEXTURE3D(_CloudNoiseHighFreq);
      float4 _CloudNoiseHighFreq_ST;
      SAMPLER(sampler_CloudNoiseHighFreq);

      TEXTURE2D(_CurlTex);
      SAMPLER(sampler_CurlTex);
      float4 _CurlTex_TexelSize;
      
      TEXTURE2D(_DensityErosionTex);
      SAMPLER(sampler_DensityErosionTex);

      TEXTURE2D(_CameraDepthTexture);
      SAMPLER(sampler_CameraDepthTexture);

      TEXTURE2D(_BlueNoise);
      SAMPLER(sampler_BlueNoise);
      float4 _BlueNoise_TexelSize;
      CBUFFER_END

      struct VertexInput {
        float4 positionOS : POSITION;
        float2 uv         : TEXCOORD0;
        float3 normalOS   : NORMAL;
        float4 tangentOS  : TANGENT;
      };

      struct VertexOutput {
        float4 positionCS : SV_POSITION;
        float2 uv         : TEXCOORD0;
        float4 viewVector : TEXCOORD1;
      };

      struct ProcessedProperties {
        float3 planetCenter;
        float planetCenterToCloudStart;
        float planetCenterToCloudEnd;
        float cloudLayerHeight;
      };

      static const float4 STRATUS_GRADIENT = float4(0.02, 0.05, 0.09, 0.11);
      static const float4 STRATOCUMULUS_GRADIENT = float4(0.02, 0.2, 0.48, 0.625);
      static const float4 CUMULUS_GRADIENT = float4(0.01, 0.0625, 0.78, 1.0);

      static const float maxDetailRemapping = 0.8;

      // random vectors on the unit sphere
      static const float3 RANDOM_VECTORS[] =
      {
        float3( 0.38051305,  0.92453449, -0.02111345),
        float3(-0.50625799, -0.03590792, -0.86163418),
        float3(-0.32509218, -0.94557439,  0.01428793),
        float3( 0.09026238, -0.27376545,  0.95755165),
        float3( 0.28128598,  0.42443639, -0.86065785),
        float3(-0.16852403,  0.14748697,  0.97460106)
      };
      
      float3 _LightDirection;

      float remap(float t, float a1, float b1, float a2, float b2) {
        return clamp(a2 + ((t - a1) * (b2 - a2) / (b1 - a1)), a2, b2);
      }

      float remapUnclamped(float t, float a1, float b1, float a2, float b2) {
        return a2 + ((t - a1) * (b2 - a2) / (b1 - a1));
      }

      uint intersectRaySphere(float3 rayOrigin, float3 rayDir, float3 sphereCenter, float sphereRad, out float2 t) {
        float3 l = rayOrigin - sphereCenter;
        float a = 1;
        float b = 2 * dot(rayDir, l);
        float c = dot(l, l) - sphereRad * sphereRad;
        float discriminate = b * b - 4 * a * c;

        if (discriminate < 0) {
          t.x = t.y = 0;
          return 0;
        }
        else if (abs(discriminate) - 0.00005 <= 0) {
          t.x = t.y = -0.5 * b / a;
          return 1;
        }
        else {
          float q = b > 0 ?
                    -0.5 * (b + sqrt(discriminate)) :
                    -0.5 * (b - sqrt(discriminate));
          
          float h1 = q / a;
          float h2 = c / q;
          t.x = min(h1, h2);
          t.y = max(h1, h2);
          if (t.x < 0) {
            t.x = t.y;
            if (t.x < 0) {
              return 0;
            }
            return 1;
          }
          return 2;
        }
      }

      VertexOutput vert(VertexInput i) {
        VertexOutput o;

        o.positionCS = mul(UNITY_MATRIX_MVP, float4(i.positionOS.xyz, 1));
        float3 viewVector = mul(unity_CameraInvProjection, float4(i.uv * 2 - 1, 0, -1));
        o.viewVector = mul(unity_CameraToWorld, float4(viewVector, 0));
        o.uv = i.uv;

        return o;
      }

      float2 worldPosToCloudCoords(float3 worldPos) {
        return (worldPos.xz * 2 - 0.5) / _WeatherMapScale;
      }

      float getHeightFactor(float3 pos, ProcessedProperties cloudInfo) {
        return saturate((distance(pos, cloudInfo.planetCenter) - cloudInfo.planetCenterToCloudStart) / cloudInfo.cloudLayerHeight);
      }

      float getDensityHeightGradient(float3 pos, float4 weatherMap, ProcessedProperties cloudInfo) {
        float t = _CloudHeightTest; // replace this with actually using the weather map when i change that
        float4 gradient = lerp(lerp(STRATUS_GRADIENT, STRATOCUMULUS_GRADIENT, saturate(t * 2)), lerp(STRATOCUMULUS_GRADIENT, CUMULUS_GRADIENT, saturate((t-0.5) * 2)), step(0.5, t));
        float height = getHeightFactor(pos, cloudInfo);
        return smoothstep(gradient.x, gradient.y, height) - smoothstep(gradient.z, gradient.w, height);
      }

      float getHorizonFadeFactor(float3 raymarchDir) {
        return (1 - smoothstep(0, -_FadeHorizonAngle, raymarchDir.y));
      }

      float getDistanceFactor(float3 pos) {
        float distance = length(pos.xz-_WorldSpaceCameraPos.xz);
        return saturate((distance - _FadeMinDistance) / _FadeMaxDistance);
      }

      float4 sampleWeatherMap(float3 pos, ProcessedProperties cloudInfo) {
        float distanceFactor = getDistanceFactor(pos);
        float2 coords = worldPosToCloudCoords(pos);
        float4 weather = SAMPLE_TEXTURE2D(_WeatherMap, sampler_WeatherMap, coords * _WeatherMap_ST.xy + _WeatherMap_ST.zw + _Time.x * float2(0.41, 0.91) * 0.1 * _WeatherMapScrollingSpeed);
        float4 result = weather;
        result.r = remap(result.r * _CloudCoverageMultiplier, 0, 1, _CloudCoverageMinimum, 1);
        result.r = min(result.r, 1 - distanceFactor);
        result.b = saturate(result.b * _CloudTypeMultiplier);
        return result;
      }

      float2 sampleHeightDensityErosion(float3 pos, float cloudType, ProcessedProperties cloudInfo)  {
        float2 heightDensityErosion = SAMPLE_TEXTURE2D(_DensityErosionTex, sampler_DensityErosionTex, float2(cloudType, getHeightFactor(pos, cloudInfo))).xy;
        return heightDensityErosion;
      }

      float FBM(float3 layers) {
        return (layers.x * 0.625) + (layers.y * 0.25) + (layers.z * 0.125);
      }

      float sampleCloudDensity(float3 worldPos, float4 weather, float heightFactor, float lod, bool highquality, ProcessedProperties cloudInfo) {
        float4 lowFreqNoises = SAMPLE_TEXTURE3D(_CloudNoiseLowFreq, sampler_CloudNoiseLowFreq, worldPos*0.001*_CloudShapeNoiseScale+float3(0.41, 0, 0.91) * 0.1 * _LowNoiseScrollingSpeed * _Time.x);
      
        float lowFreqFBM = FBM(lowFreqNoises.gba);

        float baseCloud = remap(lowFreqNoises.r, (1-lowFreqFBM), 1, 0, 1);

        float2 heightDensityErosion = sampleHeightDensityErosion(worldPos, weather.b, cloudInfo);

        baseCloud *= heightDensityErosion.x;

        // float densityHeightGradient = getDensityHeightGradient(worldPos, weather, cloudInfo);

        // baseCloud *= densityHeightGradient;

        float cloudCoverage = baseCloud - (1 - weather.r);

        cloudCoverage *= lerp(1, weather.r, _CloudDensityCoverageMultiplier * step(0, cloudCoverage));

        // float baseCloudWithCoverage = remap(baseCloud, 1-weather.r, 1, 0, 1);

        // baseCloudWithCoverage *= weather.r;

        float3 curlNoise = SAMPLE_TEXTURE2D(_CurlTex, sampler_CurlTex, worldPos.xz*0.001*_CloudShapeNoiseScale*_CurlScale).rgb;
        float3 posWithCurl = worldPos + curlNoise * (1 - heightFactor) * _CurlStrength;

        float density;

        if (highquality) {
          float3 highFreqNoises = SAMPLE_TEXTURE3D(_CloudNoiseHighFreq, sampler_CloudNoiseHighFreq, posWithCurl*0.001*_CloudShapeNoiseScale*_DetailNoiseScale+float3(0.41, 0, 0.91) * 0.1 * _HighNoiseScrollingSpeed * _Time.x).rgb;

          float highFreqFBM = FBM(highFreqNoises);

          highFreqFBM = lerp(1 - highFreqFBM, highFreqFBM, heightDensityErosion.y);

          float detailAmount = min(maxDetailRemapping, highFreqFBM * _DetailStrength);

          density = remap(cloudCoverage, detailAmount, 1, 0, 1);
        }
        else {
          density = saturate(cloudCoverage);
        }

        return density;
      }

      float beerLambert(float sampleDensity, float precipitation) {
        return exp(-sampleDensity * precipitation * (1/_BeerLambertStrength)) * 2;
      }

      float henyeyGreenstein(float lightDotView, float g) {
        float g2 = g * g;
        return ((1 - g2) / pow((1 + g2 - 2 * g * lightDotView), _SilverLiningPower)) * 0.25;
      }

      float powderSugarEffect(float sampleDensity, float lightDotView) {
        float powd = 1 - exp(-sampleDensity);
        return lerp(_PowderSugarStrength, powd, saturate((-lightDotView * 0.5) + 0.5));
      }

      float lightEnergy(float lightDotView, float densitySample, float originalDensity, float precipitation) {
        return 2 * beerLambert(densitySample, precipitation) * powderSugarEffect(originalDensity, lightDotView) * henyeyGreenstein(lightDotView, -0.8f);
      }

      float3 ambientLight(float heightFactor) {
        return lerp(_AmbientBottomColor, _AmbientTopColor, heightFactor);
      }

      float sampleCloudDensityAlongCone(float3 worldPos, float stepSize, float startOffset, float lightDotView, float startDensity, ProcessedProperties cloudInfo) {
        float3 pos = worldPos;
        float coneRadius = 1;
        float densityAlongCone = 0;
        float3 lightStep = _MainLightPosition.xyz * stepSize;
        const static float RCPLIGHTRAYITERATIONS = 1/6.0;
        float rcpThickness = 1 / (stepSize * 6);
        float density = 0;
        float4 weather = 0;

        for (int i = 0; i < 6; ++i) {
          float3 conePos = pos + startOffset + coneRadius * RANDOM_VECTORS[i] * float(i + 1);
          float heightFactor = getHeightFactor(conePos, cloudInfo);
          if (heightFactor <= 1) {
            weather = sampleWeatherMap(conePos, cloudInfo);
            float cloudDensity = sampleCloudDensity(conePos, weather, heightFactor, 0, false, cloudInfo);

            if (cloudDensity > 0) {
              density += cloudDensity;
              float transmittance = 1 - (density * rcpThickness);
              densityAlongCone += (cloudDensity * transmittance);
            }
          }
          pos += lightStep;
          coneRadius += 6;
        }

        return saturate(lightEnergy(lightDotView, densityAlongCone, startDensity, _LightAbsorbtion+1));
      }

      half4 traceClouds(float3 rayDirection, float2 screenPos, float3 startPos, float3 endPos, ProcessedProperties cloudInfo) {

        float3 dir = endPos - startPos;
        float thickness = length(dir);
        float rcpThickness = 1.0 / thickness;
        uint sampleCount = lerp(64, 128, saturate((thickness - cloudInfo.cloudLayerHeight) / cloudInfo.cloudLayerHeight));
        float stepSize = thickness / float(sampleCount);

        float startOffset = /*stepSize * */(SAMPLE_TEXTURE2D(_BlueNoise, sampler_BlueNoise, screenPos*(_ScreenParams.y*_BlueNoise_TexelSize.y)).r - 0.5) * _BlueNoiseStrength;
        
        dir /= thickness;
        float3 posStep = stepSize * dir;

        startPos += startOffset * dir;
        endPos += startOffset * dir;

        float lightDotView = -dot(normalize(_MainLightPosition.xyz), normalize(rayDirection));

        float3 pos = startPos;
        float4 weather = 0;
        float4 result = 0;

        float density = 0;
        // cloudtest dictates the quality of the cloud we sample
        float cloudTest = 0;
        int zeroDensitySampleCount = 0;

        [loop]
        for (uint i = 0; i < sampleCount; ++i) {
          float heightFactor = getHeightFactor(pos, cloudInfo);
          weather = sampleWeatherMap(pos, cloudInfo);

          // float cloudDensity = sampleCloudDensity(
          //   pos,
          //   weather,
          //   heightFactor,
          //   0,
          //   cloudInfo
          // );
          // density += cloudDensity;
          // pos += posStep;

          if (cloudTest > 0) {
            // this should be a high quality sample
            float cloudDensity = sampleCloudDensity(
              pos,
              weather,
              heightFactor,
              0,
              true,
              cloudInfo);
            
            if (cloudDensity < 0.00001) {
              zeroDensitySampleCount++;
            }

            if (zeroDensitySampleCount < 6) {
              density += cloudDensity;
              if (cloudDensity > 0) {
                float transmittance = 1 - (density * rcpThickness);
                float densityAlongCone = sampleCloudDensityAlongCone(
                  pos,
                  stepSize,
                  startOffset,
                  lightDotView,
                  cloudDensity,
                  cloudInfo);

                float3 ambientBadApprox = ambientLight(heightFactor) * min(1, length(_MainLightColor.rgb * 0.125)) * transmittance * 4;
                
                float4 source = float4(saturate((_MainLightColor.rgb * densityAlongCone * 2) + ambientBadApprox), saturate(cloudDensity * transmittance * _CloudChunkiness));
                source.rgb *= source.a;
                result = (1 - result.a) * source + result;
                if (result.a >= 1) break;
              }
              pos += posStep;
            }
            else {
              cloudTest = 0;
              zeroDensitySampleCount = 0;
            }
          }
          else {
            cloudTest = sampleCloudDensity(
              pos,
              weather,
              heightFactor,
              0,
              false,
              cloudInfo);

            if (cloudTest == 0) {
              pos += posStep;
            }
          }
        }

        float fogAmt = (1 - exp(-distance(startPos, _WorldSpaceCameraPos) * 0.0001))*result.a;
        float3 fogColor = _FogColor * length(_MainLightColor.rgb) * 0.8;
        float3 sunColor = normalize(_MainLightColor.rgb) * 4 * length(_MainLightColor.rgb);
        fogColor = lerp(fogColor, sunColor, pow(saturate(lightDotView), 8));
        // return float4(clamp(lerp(result.rgb, fogColor, fogAmt), 0, 1000), saturate(result.a));
        return result;
      }

      half4 traceSpheres(float3 rayDirection, float2 screenPos, float3 startPos, float3 endPos, ProcessedProperties cloudInfo) {
        float3 dir = endPos - startPos;
        float thickness = length(dir);
        uint sampleCount = lerp(128, 128, saturate((thickness - cloudInfo.cloudLayerHeight) / cloudInfo.cloudLayerHeight));
        float stepSize = thickness / float(sampleCount);
        
        float radius = 20;
        float tiling = 100;

        dir /= thickness;
        float3 posStep = stepSize * dir;
        
        float startOffset = /*stepSize * */(SAMPLE_TEXTURE2D(_BlueNoise, sampler_BlueNoise, screenPos*(_ScreenParams.y*_BlueNoise_TexelSize.y)).r - 0.5) * _BlueNoiseStrength;
        startPos += startOffset * dir;
        endPos += startOffset * dir;

        float3 pos = startPos;

        [loop]
        for (uint i = 0; i < sampleCount; ++i) {
          float3 posTiled = frac(pos / tiling) * tiling;
          float sphereVal = step(distance(posTiled, float3(tiling/2, tiling/2, tiling/2)), radius);
          if (sphereVal > 0 && pos.y < 500) return half4(1, (pos.y - 430)/(radius*2), 0, 1);

          pos += posStep;
        }
        return (0, 0, 0, 0);
      }

      half4 frag(VertexOutput i) : SV_TARGET {
        half4 clouds;

        ProcessedProperties cloudInfo;

        cloudInfo.planetCenter = float3(_WorldSpaceCameraPos.x, -_PlanetRadius, _WorldSpaceCameraPos.z);
        cloudInfo.planetCenterToCloudStart = _PlanetRadius + _CloudLayerStart;
        cloudInfo.planetCenterToCloudEnd = _PlanetRadius + _CloudLayerEnd;
        cloudInfo.cloudLayerHeight = _CloudLayerEnd - _CloudLayerStart;

        float rawDepth = SAMPLE_TEXTURE2D(_CameraDepthTexture, sampler_CameraDepthTexture, i.uv).r;
        bool depthPresent = rawDepth > 0;
        float depth = LinearEyeDepth(rawDepth, _ZBufferParams);

        float3 worldPos = ComputeWorldSpacePosition(i.uv, 0.5, UNITY_MATRIX_I_VP);
        float3 sceneWorldPos = ComputeWorldSpacePosition(i.uv, rawDepth, UNITY_MATRIX_I_VP);
        float3 viewDir = normalize(i.viewVector.xyz);

        float2 screenPos = float2((i.uv.x * _ScreenParams.x/_ScreenParams.y), i.uv.y);

        float2 ph = 0;
        uint planetHits = intersectRaySphere(
          _WorldSpaceCameraPos,
          viewDir,
          cloudInfo.planetCenter,
          _PlanetRadius,
          ph);
        
        float2 ih = 0;
        uint innerShellHits = intersectRaySphere(
          _WorldSpaceCameraPos,
          viewDir,
          cloudInfo.planetCenter,
          cloudInfo.planetCenterToCloudStart,
          ih);
        
        float2 oh = 0;
        uint outerShellHits = intersectRaySphere(
          _WorldSpaceCameraPos,
          viewDir,
          cloudInfo.planetCenter,
          cloudInfo.planetCenterToCloudEnd,
          oh);
        
        float3 planetHit = _WorldSpaceCameraPos + (viewDir * ph.x);
        float3 innerShellHit = _WorldSpaceCameraPos + (viewDir * ih.x);
        float3 outerShellHit = _WorldSpaceCameraPos + (viewDir * oh.x);

        // where in the shells are we?
        float distanceFromPlanetCenter = distance(_WorldSpaceCameraPos, cloudInfo.planetCenter);
        if (distanceFromPlanetCenter < cloudInfo.planetCenterToCloudStart) {
          if ((depthPresent && (distance(sceneWorldPos, _WorldSpaceCameraPos) < distance(innerShellHit, _WorldSpaceCameraPos))) || planetHits > 0) {
            clouds = half4(0, 0, 0, 0);
          }
          else if (depthPresent && (distance(sceneWorldPos, _WorldSpaceCameraPos) < distance(outerShellHit, _WorldSpaceCameraPos))){
            clouds = traceClouds(viewDir, screenPos, innerShellHit, sceneWorldPos, cloudInfo);
          }
          else {
            clouds = traceClouds(viewDir, screenPos, innerShellHit, outerShellHit, cloudInfo);
          }
        }
        else if (distanceFromPlanetCenter > cloudInfo.planetCenterToCloudEnd) {
          // this is either
          // 1 - enter outer shell, leave inner shell
          // 2 - enter outer shell, leave outer shell
          float3 firstShellHit = outerShellHit;
          if (outerShellHits == 0 || depthPresent && (distance(sceneWorldPos, _WorldSpaceCameraPos) < distance(firstShellHit, _WorldSpaceCameraPos))) {
            clouds = half4(0, 0, 0, 0);
          }
          else {
            float3 secondShellHit = outerShellHits == 2 && innerShellHits == 0 ? _WorldSpaceCameraPos + (viewDir * oh.y) : innerShellHit;
            clouds = traceClouds(viewDir, screenPos, firstShellHit, secondShellHit, cloudInfo);
          }
        }
        else {
          // we in the clouds
          float3 shellHit = innerShellHits > 0 ? innerShellHit : outerShellHit;
          if (depthPresent && (distance(sceneWorldPos, _WorldSpaceCameraPos) < distance(shellHit, _WorldSpaceCameraPos))) {
            shellHit = sceneWorldPos;
          }
          clouds = traceClouds(viewDir, screenPos, _WorldSpaceCameraPos, shellHit, cloudInfo);
        }

        half3 screen = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv);
        half4 color = half4(clouds.rgb + screen.rgb * (1 - clouds.a), 1);
        return color;
      }

      ENDHLSL
    }

  }
  FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
