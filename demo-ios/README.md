# iPad / iPhone test app for msplat 2DGS training

A minimal SwiftUI app for running on-device 2DGS training on iPad Pro M2
(primary target) or iPhone 14 Pro (memory-constrained, harder).

## Prerequisites

- macOS with Xcode 16+ (we test on Xcode 26.4.1 / SDK 26.4)
- An Apple Developer account configured in Xcode (for code signing to device)
- iPad Pro M2 (or any iPad on iOS 17+) connected via USB-C, or USB-C+Lightning for iPhone
- The XCFramework already built: `external/msplat/MsplatCore.xcframework/`
  - Build with: `./scripts/build-xcframework.sh` from the msplat repo root (takes ~5 min)

## Setup (one-time, ~5 minutes)

1. Open Xcode → File → New → Project → iOS → App.
2. Product name: `IpadTestApp`. Interface: SwiftUI. Language: Swift. Save it
   anywhere on your Mac (NOT inside the msplat repo — keep them separate).
3. With the new project open in Xcode:
   - **Add the Msplat package as a local dependency.**
     File → Add Package Dependencies → Add Local…
     Pick `<path-to-Facescan>/external/msplat/swift/` (the `Package.swift` lives there).
     Add to the `IpadTestApp` target.
   - **Replace the auto-generated `ContentView.swift` + `IpadTestApp.swift`** with the
     single file at `demo-ios/Sources/IpadTestApp/IpadTestApp.swift`. (Drag it in,
     or copy-paste the contents.)
   - **Bundle the dataset.** In Finder, locate `~/Downloads/Scan_recent/`.
     Drag the entire folder into the Xcode project navigator. In the import sheet:
       - **Action**: "Create folder references" (the folder turns blue, NOT yellow).
       - **Add to targets**: tick `IpadTestApp`.
     The Scan_recent folder should now appear in the navigator as a blue folder.
4. **Code signing.** Select the project in the navigator → `IpadTestApp` target →
   Signing & Capabilities. Check "Automatically manage signing" and pick your team.
   Set Bundle Identifier to something unique like `com.yourname.IpadTestApp`.
5. **Pick the iPad as run destination.** Connect iPad via USB. In Xcode's toolbar,
   set the destination to your iPad (not Simulator).
6. **Trust this developer on the iPad** (first time only):
   On the iPad → Settings → General → VPN & Device Management → tap your profile →
   "Trust".

## Run

7. Hit ⌘R in Xcode. The app builds, installs to the iPad, and launches.
8. Tap **Start**. Default is 200 iterations. The progress bar advances, loss
   value updates every 10 iterations, and a live render of camera 0 displays.
9. Expected output (200 iters from random init):
   - Loss drops from ~0.22 → ~0.06 over 200 steps
   - Splat count stays ~100,000 (densify hasn't kicked in at low step counts)
   - ms/step around 200–500 on iPad Pro M2 (vs ~250 on macOS dev machine)
   - Resident memory ~500–800 MB

## Troubleshooting

- **"Scan_recent folder not in app bundle" warning shown in the app.**
  The dataset folder didn't make it into the bundle as a folder reference.
  Re-add via Xcode's File → Add Files… and pick "Create folder references".
- **Build error: "MsplatCore module not found".**
  The XCFramework wasn't generated. Run `./scripts/build-xcframework.sh` from
  the msplat repo root, then clean+rebuild in Xcode.
- **Build error: "iOS deployment target".**
  The Msplat package targets iOS 17+. In Xcode, IpadTestApp target → General →
  Minimum Deployments → iOS 17.0.
- **App crashes immediately on launch / Metal error.**
  Check the iPad is on iOS 17 or newer. Older iOS versions don't have MSL 3.0
  atomic_float support which the backward kernels rely on.
- **Memory pressure / app gets killed mid-training on iPhone.**
  Reduce iterations (the stepper at the bottom). iPhone 14 Pro has ~3 GB
  usable; 100k splats at 1280×720 input is near that limit. Try downscaling
  the dataset or using a smaller subset of frames.

## What this is testing

This app exercises the full Phase 2b Milestone 2 stack on a real Apple GPU
that isn't the macOS dev machine — making sure:
1. The Metal kernels (forward + backward + Adam) work on iPad / iPhone GPU family.
2. The MSL 3.0 atomic_float operations behave correctly on mobile.
3. Memory pressure is manageable at face-scan splat counts (~100k).
4. End-to-end training reaches loss-decrease parity with macOS.

If all four pass, Phase 2b Milestone 3 (device validation) is done.
