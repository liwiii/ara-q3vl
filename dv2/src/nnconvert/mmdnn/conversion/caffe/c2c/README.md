# C2C PASS CONFIG
C2C patches are run in transform.py based on where they are added in TransformConfig.yaml file.<br />
<br />
There are four kind of patchings<br />
    [1] FinalPatchings<br />
    [2] LayerSpecificPatchings<br />
    [3] NamePatchings<br />
    [4] Standardize<br />
<br />
Other than this the TransformConfig.yaml also contains patchings to ignore for a target platform like Ara1, Ara2, Tflite<br />
<br />

## How to add Patchings into config
Suppose the patching is InnerProductPatching and it is written in from mmdnn.conversion.caffe.c2c.inrp
and let us suppose Layer 'Inrp' needs this patching.<br />
<br />
[1] In the TransformConfig.yaml file, there are Module, Patching and Ignore.<br />
[2] Under Patching, add Inrp:<br />
[3] Under Inrp:, add - InnerProductPatching<br />
    Example,<br />

```
      Patching:
        Inrp:
        - InnerProductPatching
```
<br />
[4] Now in Module, add inrp(based on mmdnn.conversion.caffe.c2c.inrp) in same location<br />

```
      Module:
        Inrp:
        - inrp
```
<br />
Modules specifies the path to import from and Patchings provide what to import<br />
<br />
For Finalpatchings etc there is no layer name and we would add all of them directly<br />
Example,<br />

```
    Patching:
      FinalPatchings:
      - InnerProductPatching
```
<br />

<br />

```
    Module:
      FinalPatchings:
      - InnerProductPatching
```
<br />

## Ignore Section
For any given target, say Ara1, all patchings that start with prefixes mentioned under it in ignore will be ignored.<br />
<br />
For example,<br />

```
    Ignore:
      Ara1:
      - Inner
```
<br />
This would ignore patchings like InnerProductPatching for Ara1 target.<br />
