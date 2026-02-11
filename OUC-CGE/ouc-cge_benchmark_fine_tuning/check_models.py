import timm
print("--- Checking all available model architectures ---")
all_models = timm.list_models('*uniformer*')
if not all_models:
    print("No 'uniformer' models found in the library source code.\n")
else:
    for model_name in all_models:
        print(model_name)

print("\n--- Checking for PRETRAINED ResNet models (tests network access) ---")
pretrained_resnet_models = timm.list_models('*resnet*', pretrained=True)
if not pretrained_resnet_models:
    print("Could not find any pretrained ResNet models. This points to a network or cache issue.")
else:
    print(f"Successfully found {len(pretrained_resnet_models)} pretrained ResNet models.")