# Wild-4K

Wild-4K is the in-the-wild evaluation set behind the Tracking Success numbers in the
[Model Zoo](../../README.md#model-zoo). It has 4182 clips selected from
[Koala-36M-v1](https://huggingface.co/datasets/Koala-36M/Koala-36M-v1).

We release only the clip list; videos are not distributed.

## Clip list

`wild4k.csv` has one clip per row, sorted by ID:

```
id,category
-0jUX5ZtWJ8_97,normal
-16lClEdFKg_24,self_occluded
...
```

- `id`: the `videoID` in the Koala-36M-v1 metadata, of the form
  `<youtube_id>_<clip_index>`, e.g. `03gz-VVANRE_15` is clip 15 of source video `03gz-VVANRE`.
- `category`, labeled per frame by a vision-language model:

| Category | Clips | Description |
|----------|------:|-------------|
| `self_occluded` | 1384 | the body hides many of its own key parts in some frames (back to the camera, strong side view, limbs or torso blocking each other) |
| `hard` | 669 | high-difficulty action without large self-occlusion (yoga, gymnastics, handstand, flip, split, push-up, extreme flexibility or strength poses) |
| `normal` | 2129 | everything else (walking, running, dancing, ball and racket sports, ...) |

A clip with both large self-occlusion and a high-difficulty action is labeled `self_occluded`.

## Preprocessing

The reported results use the first 10 seconds of each clip at 30 fps, without audio:

```bash
ffmpeg -i <id>.mp4 -t 10 -vf fps=30 -an -c:v libx264 -pix_fmt yuv420p -crf 18 <id>_30fps.mp4
```

## License

The clips are subject to the Koala-36M-v1 terms and those of their original sources;
see the [dataset page](https://huggingface.co/datasets/Koala-36M/Koala-36M-v1).
