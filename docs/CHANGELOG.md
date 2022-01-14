# Release Notes

## 0.0.8 - 2022-01-14

* 🤩 refactor core by [@aniketmaurya] in [#136](https://github.com/gradsflow/gradsflow/pull/136)
* cleanup APIs by [@aniketmaurya] in [#137](https://github.com/gradsflow/gradsflow/pull/137)
* added conda installation instruction by [@sugatoray] in [#144](https://github.com/gradsflow/gradsflow/pull/144)
* recursively exclude tests folder and its contents by [@sugatoray] in [#141](https://github.com/gradsflow/gradsflow/pull/141)
* add model.save test by [@aniketmaurya] in [#147](https://github.com/gradsflow/gradsflow/pull/147)
* remove redundant to_item by [@aniketmaurya] in [#152](https://github.com/gradsflow/gradsflow/pull/152)
* refactor Tracker by [@aniketmaurya] in [#153](https://github.com/gradsflow/gradsflow/pull/153)
* Change methods not using its bound instance to staticmethods by [@deepsource-autofix]
  in [#156](https://github.com/gradsflow/gradsflow/pull/156)
* refactor metrics by [@aniketmaurya] in [#159](https://github.com/gradsflow/gradsflow/pull/159)
* add dataoader length by [@aniketmaurya] in [#160](https://github.com/gradsflow/gradsflow/pull/160)
* fix model checkpoint folder not found by [@aniketmaurya] in [#162](https://github.com/gradsflow/gradsflow/pull/162)
* Fix metrics update by [@aniketmaurya] in [#163](https://github.com/gradsflow/gradsflow/pull/163)
* Replace multiple `==` checks with `in` by @deepsource-autofix in [#167](https://github.com/gradsflow/gradsflow/pull/167)
* increment current_epoch after each epoch by [@aniketmaurya] in [#169](https://github.com/gradsflow/gradsflow/pull/169)
* Wandb Implementation by [@aniketmaurya] in [#168](https://github.com/gradsflow/gradsflow/pull/168)

**Full Changelog**: https://github.com/gradsflow/gradsflow/compare/v0.0.7...v0.0.8

## 0.0.7 - 2021-11-26

* ☄️ comet integration [#129](https://github.com/gradsflow/gradsflow/pull/129)
* add model checkpoint callback [#121](https://github.com/gradsflow/gradsflow/pull/121)
* 📝 add csv logger [#116](https://github.com/gradsflow/gradsflow/pull/116)
* 🚀 add train_eval_callback [#111](https://github.com/gradsflow/gradsflow/pull/111)
* 🪄 add Average Meter [#109](https://github.com/gradsflow/gradsflow/pull/109)
* fix device issue in metric calculation PR [#106](https://github.com/gradsflow/gradsflow/pull/106)

## 0.0.6 - 2021-10-4

* 🎉 Revamp Callbacks and Training. PR [#94](https://github.com/gradsflow/gradsflow/pull/94)
* ✨ refactor data handling 📝 docs update. PR [#91](https://github.com/gradsflow/gradsflow/pull/91)
* integrate torchmetrics. PR [#80](https://github.com/gradsflow/gradsflow/pull/80)
* callbacks & 🤑 ProgressCallback. PR [#76](https://github.com/gradsflow/gradsflow/pull/76)
* 🔥 Add AutoModel Tuner. PR [#74](https://github.com/gradsflow/gradsflow/pull/74)
* refactor APIs - Simplify API & add `model.compile(...)`. PR [#73](https://github.com/gradsflow/gradsflow/pull/73)
* 🤗 integrate HF Accelerator. PR [#71](https://github.com/gradsflow/gradsflow/pull/71)

## 0.0.5 - 2021-9-26

* 🔥 Add custom training loop with `model.fit`. PR [#63](https://github.com/gradsflow/gradsflow/pull/63) Done
  by [@aniketmaurya](https://github.com/aniketmaurya)
* ☁️ Add `ray.data` - remote dataset loader. PR [#61](https://github.com/gradsflow/gradsflow/pull/61) Done
  by [@aniketmaurya](https://github.com/aniketmaurya)
* 🎉 Add AutoDataset - Encapsulate datamodule and dataloaders. PR [#59](https://github.com/gradsflow/gradsflow/pull/59)
  Done by [@aniketmaurya](https://github.com/aniketmaurya)
* 🌟 Add Autotask feature. PR [#54](https://github.com/gradsflow/gradsflow/pull/54) Done
  by [@gagan3012](https://github.com/gagan3012)
* ✨ Add AutoTrainer to support plain torch training loop and other torch frameworks.
  PR [#53](https://github.com/gradsflow/gradsflow/pull/53)

## 0.0.4 - 2021-9-3

* fix best checkpoints model loading. PR [#52](https://github.com/gradsflow/gradsflow/pull/52)
* 🚀 feature/fix train arguments docs PR [#44](https://github.com/gradsflow/gradsflow/pull/44)
* Publish Python 🐍 distributions 📦 to PyPI [#42](https://github.com/gradsflow/gradsflow/pull/42)

## 0.0.3 - 2021-8-31

* add optuna visualizations 🎨 . PR [#27](https://github.com/gradsflow/gradsflow/pull/27)
  by [@aniketmaurya](https://github.com/aniketmaurya).
* add max_steps for HPO. PR [#25](https://github.com/gradsflow/gradsflow/pull/25)
  by [@aniketmaurya](https://github.com/aniketmaurya).
* :memo: update docs & license. PR [#23](https://github.com/gradsflow/gradsflow/pull/23)
  by [@aniketmaurya](https://github.com/aniketmaurya).
* fetch best trial model. PR [#21](https://github.com/gradsflow/gradsflow/pull/21)
  by [@aniketmaurya](https://github.com/aniketmaurya).
* migrate to ray_tune 🌟. Read more [here](https://github.com/gradsflow/gradsflow/issues/35).
  PR [#36](https://github.com/gradsflow/gradsflow/pull/36) by [@aniketmaurya](https://github.com/aniketmaurya).
* render jupyter notebooks in documentation. PR [#38](https://github.com/gradsflow/gradsflow/pull/38)
  by [@aniketmaurya](https://github.com/aniketmaurya).

## 0.0.2 - 2021-8-27

* Fix max steps validation key error. PR [#31](https://github.com/gradsflow/gradsflow/pull/31)
  by [@aniketmaurya](https://github.com/aniketmaurya).

## 0.0.1 - 2021-8-25

* 📝 update example and documentation. Done by [ aniketmaurya](https://github.com/aniketmaurya). Check
  the [Pull Request 20 with the changes and stuff](https://github.com/gradsflow/gradsflow/pull/20).
* :tada::sparkles: First Release - v0.0.1 - Refactor API & tested Python 3.7+. Done
  by [ aniketmaurya](https://github.com/aniketmaurya). Check
  the [Pull Request 18 with the changes and stuff](https://github.com/gradsflow/gradsflow/pull/18).
* Adding example notebook for AutoSummarization. Done by [the GitHub user gagan3012](https://github.com/gagan3012).
  Check the [Pull Request 19 with the changes and stuff](https://github.com/gradsflow/gradsflow/pull/19).
* Adding text summarisation. Done by [the GitHub user gagan3012](https://github.com/gagan3012). Check
  the [Pull Request 14 with the changes and stuff](https://github.com/gradsflow/gradsflow/pull/14).
* add codecov CI. Done by [the GitHub user aniketmaurya](https://github.com/aniketmaurya). Check
  the [Pull Request 15 with the changes and stuff](https://github.com/gradsflow/gradsflow/pull/15).
* 📚 update documentation - added citation, acknowledgments, docstrings automation. Done
  by [the GitHub user aniketmaurya](https://github.com/aniketmaurya). Check
  the [Pull Request 13 with the changes and stuff](https://github.com/gradsflow/gradsflow/pull/13).
* Refactor API Design, CI & Docs PR [#10](https://github.com/gradsflow/gradsflow/pull/10)
  by [@aniketmaurya](https://github.com/aniketmaurya).
* auto docstring. PR [#7](https://github.com/gradsflow/gradsflow/pull/7)
  by [@aniketmaurya](https://github.com/aniketmaurya).
* Add AutoImageClassifier. PR [#1](https://github.com/gradsflow/gradsflow/pull/1)
  by [@aniketmaurya](https://github.com/aniketmaurya).
