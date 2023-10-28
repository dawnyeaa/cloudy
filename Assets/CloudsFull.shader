Shader "Custom/CloudsFull" {
  Properties {
    _MainTex ("Texture", 2D) = "white" {}

    _PlanetRadius ("Planet Radius", Float) = 6000000
    _CloudLayerStart ("Cloud Layer Start", Float) = 400
    _CloudLayerEnd ("Cloud Layer End", Float) = 1000

    [HDR] _AmbientBottomColor ("Bottom Ambient Color", Color) = (0.5, 0.67, 0.82, 1)
    [HDR] _AmbientTopColor ("Top Ambient Color", Color) = (1, 1, 1, 1)

    _CloudCoverage ("Coverage", Range(0, 1)) = 0.35
    _CloudType ("Cloud Type", Range(0, 1)) = 0.5
    _CloudWetness ("Cloud Wetness", Range(0, 1)) = 0

    _CloudShapeNoiseScale ("Shape Noise Scale", Float) = 1

    _WeatherMap ("Weather Map", 2D) = "white" {}

    _DetailStrength ("Detail Strength", Float) = 1
    _DetailNoiseScale ("Detail Noise Scale", Float) = 1

    _FadeMinDistance ("Cloud Min Fade Distance", Float) = 10000
    _FadeMaxDistance ("Cloud Max Fade Distance", Float) = 20000

    _CloudNoiseLowFreq ("Low Frequency Cloud Noise", 3D) = "white" {}

    _CloudNoiseHighFreq ("High Frequency Cloud Noise", 3D) = "white" {}

    _CurlTex ("Curl Texture", 2D) = "white" {}
    _CurlStrength ("Curl Strength", Float) = 1
    _CurlScale ("Curl Scale", Float) = 1

    _BlueNoise("Blue Noise", 2D) = "white" {}
    _BlueNoiseStrength("Blue Noise Strength", Float) = 1

    _A ("A", Float) = 1
    _B ("B", Float) = 0

    _Extinction ("Extinction", Range(0, 10)) = 1
    _ExtinctionNear ("Extinction Local", Range(0, 10)) = 1.5
    _ExtinctionFar ("Extinction Global", Range(0, 10)) = 2
    _ExtinctionColorBlend ("Extinction Blend", Range(0, 1)) = 1
    _ExtinctionColorScalar ("Extinction Scalar", Range(0, 2)) = 0.8
    _Scattering ("Scattering", Range(0, 100)) = 100
    _PowderNear ("Powder Local", Range(0, 10)) = 1.1
    _PowderFar ("Powder Global", Range(0, 10)) = 1.1
    _PowderCoefNear ("Powder Coef Local", Range(0.5, 10)) = 3
    _PowderCoefFar ("Powder Coef Global", Range(0.5, 10)) = 2
    _BottomDarkeningStart ("Bottom Darkening Start", Range(0, 1)) = 0.5
    _BottomDarkening ("Bottom Darkening", Range(0, 1)) = 0.7
    _ExtinctionReducer ("Top Extinction Reducer", Range(0, 1)) = 1
    _DensityEndPos ("Local Density End Pos", Range(0, 1)) = 0.4
    _ScatteringColorBlend ("Scattering Anti Color Blend", Range(0, 10)) = 0.5

    _LightSampleStepSize ("Light Sample Step Size", Float) = 1

    _DetailErosionLow ("Detail Erosion Low", Range(0, 1)) = 0.3
    _DetailErosionHigh ("Detail Erosion High", Range(0, 1)) = 0.25

    _ExtinctionColor ("Extinction Color", Color) = (0.5, 0.5, 0.5, 1)
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
      float _PlanetRadius;
      float _CloudLayerStart;
      float _CloudLayerEnd;

      float _CloudShapeNoiseScale;
      float _CloudCoverage;
      float _CloudType;
      float _CloudWetness;

      float3 _AmbientBottomColor;
      float3 _AmbientTopColor;

      float _DetailStrength;
      float _DetailNoiseScale;

      float _CurlStrength;
      float _CurlScale;

      float _FadeMinDistance;
      float _FadeMaxDistance;

      float _BlueNoiseStrength;

      float _A, _B;

      float _Extinction;
      float _ExtinctionNear;
      float _ExtinctionFar;
      float _ExtinctionColorBlend;
      float _ExtinctionColorScalar;
      float _Scattering;
      float _PowderNear;
      float _PowderFar;
      float _PowderCoefNear;
      float _PowderCoefFar;
      float _BottomDarkeningStart;
      float _BottomDarkening;
      float _ExtinctionReducer;
      float _DensityEndPos;
      float _DetailErosionLow;
      float _DetailErosionHigh;
      float _ScatteringColorBlend;

      float _LightSampleStepSize;

      float3 _ExtinctionColor;
      
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

      // static const float4 STRATUS_GRADIENT = float4(0.02, 0.05, 0.09, 0.11);
      // static const float4 STRATOCUMULUS_GRADIENT = float4(0.02, 0.2, 0.48, 0.625);
      // static const float4 CUMULUS_GRADIENT = float4(0.01, 0.0625, 0.78, 1.0);
      static const float2 STRATOCUMULUS_GRADIENT = float2(0.1, 0.3);
      static const float2 CUMULUS_GRADIENT = float2(0.2, 1.0);
      static const float2 CUMULONIMBUS_GRADIENT = float2(0.75, 1.0);

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

      float setRangeClamped(float t, float a, float b) {
        // return saturate((t - a)/(b - a));
        float v = clamp(t, a, b);
        v = (t - a)/(b - a);
        return saturate(v);
      }

      float exponential_integral(float z) {
        return 0.5772156649015328606065 + log( 1e-4 + abs(z) ) + z * (1.0 + z * (0.25 + z * ( (1.0/18.0) + z * ( (1.0/96.0) + z * (1.0/600.0) ) ) ) ); // For x!=0
      }

      float luminance(float3 c) {
        return sqrt(0.299 * c.r * c.r + 0.587 * c.g * c.g + 0.114 * c.b * c.b);
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

      float2 getCloudGradient(float cloudType) {
        float a = 1 - saturate(cloudType * 2);
        float b = 1 - abs(cloudType - 0.5) * 2;
        float c = saturate(cloudType - 0.5) * 2;

        return (STRATOCUMULUS_GRADIENT * a) + (CUMULUS_GRADIENT * b) + (CUMULONIMBUS_GRADIENT * c);
      }

      float getAltitudeScalar(float cloudType) {
        return lerp(8, 2, cloudType);
      }

      float getDetailErosion(float coverage) {
        float2 s = saturate(coverage * 5 + float2(0, 0.5));
        return float2(_DetailErosionLow, _DetailErosionHigh) * s;
      }

      float getAltitudeCoverage(float altitude, float2 gradient) {
        float coverage1 = saturate(altitude * 10);
        float coverage2 = 1 - saturate((altitude - gradient.x) / (gradient.y - gradient.x));
        return coverage1 * coverage2;
      }

      float getDistanceFactor(float3 pos) {
        float distance = length(pos.xz-_WorldSpaceCameraPos.xz);
        return saturate((distance - _FadeMinDistance) / _FadeMaxDistance);
      }

      float FBM(float3 layers) {
        return (layers.x * 0.625) + (layers.y * 0.25) + (layers.z * 0.125);
      }

      float sampleLowCloud(float3 worldPos, float altitude, float altitudeScalar, float gradientValues, float viewDependentLerp, float cloudCoverage, float fadeTerm, float erodeLow) {
        float altitudeCoverage = getAltitudeCoverage(altitude, gradientValues);

        float extraCoverage = (1 - saturate(altitude * altitudeScalar)) * 0.5;
        extraCoverage = lerp(0, extraCoverage, pow(viewDependentLerp, 2));

        float a = 1 - fadeTerm;
        float c = min(cloudCoverage, a);

        float lowRange = lerp(1, (c * _A + _B) - extraCoverage, altitudeCoverage);
        lowRange = saturate(lowRange);

        float4 tex = SAMPLE_TEXTURE3D(_CloudNoiseLowFreq, sampler_CloudNoiseLowFreq, worldPos*0.001*_CloudShapeNoiseScale);

        float cloudsValue = setRangeClamped(tex.r, lowRange, 1.0);
        cloudsValue = setRangeClamped(cloudsValue, tex.g * erodeLow, 1.0);
        cloudsValue = setRangeClamped(cloudsValue, tex.b * erodeLow, 1.0);
        cloudsValue = setRangeClamped(cloudsValue, tex.a * erodeLow, 1.0);

        return cloudsValue;
      }

      float3 getAmbientColor(float altitude, float extinction) {
        float ambientTerm = -extinction * saturate(1 - altitude);
        float3 isotropicScatteringTop = _AmbientTopColor * max(0, exp(ambientTerm) - ambientTerm * exponential_integral(ambientTerm));

        ambientTerm = -extinction * altitude;
        float3 isotropicScatteringBottom = _AmbientBottomColor * max(0, exp(ambientTerm) - ambientTerm * exponential_integral(ambientTerm));
      
        isotropicScatteringTop *= saturate(altitude * 0.5);

        return isotropicScatteringTop + isotropicScatteringBottom;
      }

      float sampleLightRay(float3 worldPos, float viewDependentLerp, ProcessedProperties cloudInfo) {
        float3 pos = worldPos;
        float selfShadowingExtinction = _ExtinctionNear;
        float stepSize = _LightSampleStepSize;
        float3 lightStep = _MainLightPosition.xyz * stepSize;

        float lightSampleExtinction = 1;

        // not using weather map, manual overrides
        float4 weatherData = float4(_CloudCoverage, _CloudType, _CloudWetness, 1);

        for (int i = 1; i < 6; ++i) {
          float3 lightSamplePos = pos + lightStep * i;
          float fadeTerm = getDistanceFactor(lightSamplePos);

          float2 gradientValues = getCloudGradient(weatherData.g);

          float altitude = getHeightFactor(lightSamplePos, cloudInfo);
          float altitudeScalar = getAltitudeScalar(weatherData.g);
          float extinctionAltitudeScalar = lerp(1, _ExtinctionReducer, saturate(altitude * altitudeScalar));

          if (altitude > 1) break;

          float detailErosion = getDetailErosion(weatherData.r);

          float value = sampleLowCloud(lightSamplePos, altitude, altitudeScalar, gradientValues, viewDependentLerp, weatherData.r, fadeTerm, detailErosion);
        
          float extinctionCoeff = extinctionAltitudeScalar * selfShadowingExtinction * value;
          float beersTerm = exp(-extinctionCoeff * stepSize);

          lightSampleExtinction *= beersTerm;
        }

        return lightSampleExtinction;
      }

      half4 traceClouds(float3 rayDirection, float2 screenPos, float3 startPos, float3 endPos, ProcessedProperties cloudInfo) {

        float3 dir = endPos - startPos;
        float thickness = length(dir);
        float rcpThickness = 1.0 / thickness;
        uint sampleCount = lerp(256, 256, saturate((thickness - cloudInfo.cloudLayerHeight) / cloudInfo.cloudLayerHeight));
        float stepSize = thickness / float(sampleCount);

        float startOffset = stepSize * (SAMPLE_TEXTURE2D(_BlueNoise, sampler_BlueNoise, screenPos*(_ScreenParams.y*_BlueNoise_TexelSize.y)).r - 0.5) * _BlueNoiseStrength;
        
        dir /= thickness;
        float3 posStep = stepSize * dir;

        startPos += startOffset * dir;
        endPos += startOffset * dir;

        float lightDotView = -dot(normalize(_MainLightPosition.xyz), normalize(rayDirection));

        float3 pos = startPos;
        float4 weather = 0;
        float4 result = 0;

        float density = 0;
        int zeroDensitySampleCount = 0;

        float weightedNumSteps = 0;
        float weightedNumStepsSum = 0.000001;
        float weightedExtinctionAltitude = 1;
        float weightedExtinctionAltitudeSum = 0.000001;

        float extinction = 1;
        float3 scattering = 0;

        // not using weather map, manual overrides
        float4 weatherData = float4(_CloudCoverage, _CloudType, _CloudWetness, 1);

        [loop]
        for (uint i = 1; i < sampleCount && extinction > 0.001; ++i) {
          pos += posStep;

          float fadeTerm = getDistanceFactor(pos);
          float2 gradientValues = getCloudGradient(weatherData.g);
          float altitudeScalar = getAltitudeScalar(weatherData.g);

          float altitude = getHeightFactor(pos, cloudInfo);
          float extinctionAltitudeScalar = lerp(1, _ExtinctionReducer, saturate(altitude * altitudeScalar));

          float viewDependentLerp = saturate(length(pos.xy)/_FadeMaxDistance);

          float detailErosion = getDetailErosion(weatherData.r);

          float lowCloud = sampleLowCloud(pos, altitude, altitudeScalar, gradientValues, viewDependentLerp, weatherData.r, fadeTerm, _DetailErosionLow);
        
          if (lowCloud <= 0) continue;

          float lightExtinction = sampleLightRay(pos, viewDependentLerp, cloudInfo);

          float scatteringCoeff = _Scattering * lowCloud * lightExtinction;
          float extinctionCoeff = extinctionAltitudeScalar * _Extinction * lowCloud;

          float powderAmount = lerp(_PowderNear, _PowderFar, viewDependentLerp);
          float powderCoeff = lerp(_PowderCoefNear, _PowderCoefFar, viewDependentLerp) * lowCloud;
          float powderTerm = 1 - saturate(exp(-powderCoeff * _LightSampleStepSize * 2) * powderAmount);

          float beersTerm = exp(-extinctionCoeff * _LightSampleStepSize);
          extinction *= beersTerm;

          float3 ambientColor = getAmbientColor(altitude, extinction);
          float3 scatteringTerm = scatteringCoeff * _LightSampleStepSize * powderTerm * _MainLightColor + ambientColor * powderTerm;

          scattering += extinction * scatteringTerm;

          float densityWeight = 1 - beersTerm;
          float altitudeWeight = (sampleCount - i) * lowCloud;

          weightedNumSteps += i * densityWeight;
          weightedExtinctionAltitude += altitude * altitudeWeight;

          weightedNumStepsSum += densityWeight;
          weightedExtinctionAltitudeSum += altitudeWeight;
        }

        weightedNumSteps /= weightedNumStepsSum;
        weightedExtinctionAltitude /= weightedExtinctionAltitudeSum;
        float3 closestPos = weightedNumStepsSum < 0.001 ? endPos : startPos + dir * stepSize * weightedNumSteps;

        scattering = max(scattering, 0);

        float altitudeScalar = getAltitudeScalar(weatherData.g);

        float darkBottoms = weightedExtinctionAltitude * altitudeScalar - _BottomDarkeningStart;
        float b = lerp(_ExtinctionColorScalar * _BottomDarkening, _ExtinctionColorScalar, saturate(darkBottoms));

        float3 finalColor = scattering + _ExtinctionColor * b;

        float scatteringBlendOffset = luminance(scattering) * _ScatteringColorBlend;
        float alpha = (1 - extinction) * saturate(_ExtinctionColorBlend + scatteringBlendOffset);

        return float4(finalColor, saturate(alpha));
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
