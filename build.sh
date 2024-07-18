# bazelisk  build //nexus:nexus --verbose_failures

# bazelisk build //nexus/turing/proto:proto_text --verbose_failures
# bazelisk build //nexus/test/... --verbose_failures --config=use-lld
# bazelisk  run //nexus/cc/common:adaptor_test --verbose_failures --config=use-lld

# bazelisk  build //nexus/turing/core/... --verbose_failures --config=use-lld



# bazelisk build --override_repository=bazel_vscode_compdb=/home/yinze/.vscode-server/extensions/galexite.bazel-cpp-tools-1.0.5/compdb/ --aspects=@bazel_vscode_compdb//:aspects.bzl%compilation_database_aspect --color=no --show_result=2147483647 --noshow_progress --noshow_loading_progress --output_groups=compdb_files,header_files --build_event_json_file=/tmp/tmp-8080-E1iWiIa3ZdUZ --action_env=BAZEL_CPP_TOOLS_TIMESTAMP=1718883481.203 //nexus/test:load_and_run && /home/yinze/.vscode-server/extensions/galexite.bazel-cpp-tools-1.0.5/compdb/postprocess.py -s -b /tmp/tmp-8080-E1iWiIa3ZdUZ && rm /tmp/tmp-8080-E1iWiIa3ZdUZ

# bazelisk build //nexus/cc/...  --verbose_failures \
#     --override_repository=bazel_vscode_compdb=/home/yinze/.vscode-server/extensions/galexite.bazel-cpp-tools-1.0.5/compdb/ --aspects=@bazel_vscode_compdb//:aspects.bzl%compilation_database_aspect --color=no --show_result=2147483647 --noshow_progress --noshow_loading_progress --output_groups=compdb_files,header_files --build_event_json_file=/tmp/tmp-2178-R6q9rCL2B4RH --action_env=BAZEL_CPP_TOOLS_TIMESTAMP=1718762709.226 && /home/yinze/.vscode-server/extensions/galexite.bazel-cpp-tools-1.0.5/compdb/postprocess.py -s -b /tmp/tmp-2178-R6q9rCL2B4RH && rm /tmp/tmp-2178-R6q9rCL2B4RH



# bazelisk build //nexus  --verbose_failures --config=use-lld

# bazelisk build //nexus/test/...  --strip=never  -c dbg  --verbose_failures --config=use-lld
bazelisk build //nexus/cc/common:heap_test  --strip=never  -c dbg  --verbose_failures --config=use-lld