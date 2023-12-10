using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.VFX;

[ExecuteAlways]
public class GrassEditorTweak : MonoBehaviour {
  public VisualEffect vfx;
  public Material terrainMat;
  public Color color1, color2;
  public Color fogColor;
  void OnValidate() {
    if (terrainMat) {
      terrainMat.SetColor("_color_1", color1);
      terrainMat.SetColor("_color_2", color2);
      terrainMat.SetColor("_fog", fogColor);
    }
    if (vfx) {
      vfx.SetVector4("color 1", color1);
      vfx.SetVector4("color 2", color2);
      vfx.SetVector4("fog", fogColor);
    }
  }
}
