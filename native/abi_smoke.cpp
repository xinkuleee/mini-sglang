#include "minisgl_bridge.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(std::string(message) + ": " + msgl_error());
}
int main(int argc, char **argv) {
    try {
        check(msgl_create(nullptr) == nullptr, "invalid config must fail");
        check(std::strlen(msgl_error()) > 0, "invalid config must explain failure");
        check(msgl_vocab_size(nullptr) == -1, "invalid handle must fail");
        msgl_destroy(nullptr);
        if (argc == 1) {
            std::cout << "ABI error boundary passed\n";
            return 0;
        }
        msgl_config config{argv[1], 512, 256, 8, 2, 0};
        std::unique_ptr<msgl_handle, decltype(&msgl_destroy)> h(msgl_create(&config), msgl_destroy);
        check(h != nullptr, "model creation");
        const char *prompt = "Once upon a time";
        size_t count = 0;
        check(msgl_tokenize(h.get(), prompt, std::strlen(prompt), nullptr, 0, &count) >= 0,
              "token sizing");
        std::vector<int32_t> tokens(count);
        check(msgl_tokenize(h.get(), prompt, std::strlen(prompt), tokens.data(), tokens.size(),
                            &count) == 0,
              "tokenization");
        check(count > 0 && count < 255, "prompt fits smoke test");
        const int32_t vocab = msgl_vocab_size(h.get());
        check(vocab > 0, "vocabulary");
        std::vector<msgl_batch_token> batch;
        for (size_t i = 0; i < count; ++i)
            batch.push_back({tokens[i], static_cast<int32_t>(i), 0, i + 1 == count ? 1 : 0});
        std::vector<float> initial(vocab), shared(vocab), replayed(vocab);
        check(msgl_forward(h.get(), batch.data(), batch.size(), initial.data(), initial.size()) ==
                  0,
              "prefill");
        const int32_t next = static_cast<int32_t>(std::max_element(initial.begin(), initial.end()) -
                                                  initial.begin());
        size_t piece_size = 0;
        check(msgl_piece(h.get(), next, nullptr, 0, &piece_size) >= 0, "piece sizing");
        check(msgl_is_eog(h.get(), next) >= 0, "EOG lookup");
        check(msgl_copy_seq(h.get(), 0, 1, static_cast<int32_t>(count)) == 0, "prefix reference");
        check(msgl_copy_seq(h.get(), 0, 1, static_cast<int32_t>(count)) == -1,
              "nonempty destination must fail");
        check(msgl_remove_seq(h.get(), 0) == 0, "source release");
        msgl_batch_token decode{next, static_cast<int32_t>(count), 1, 1};
        check(msgl_forward(h.get(), &decode, 1, shared.data(), shared.size()) == 0,
              "decode through shared prefix");
        check(msgl_remove_seq(h.get(), 1) == 0, "shared release");
        for (auto &item : batch) {
            item.sequence = 2;
            item.logits = 0;
        }
        decode.sequence = 2;
        batch.push_back(decode);
        check(msgl_forward(h.get(), batch.data(), batch.size(), replayed.data(), replayed.size()) ==
                  0,
              "uncached replay");
        float maximum_difference = 0;
        for (size_t i = 0; i < shared.size(); ++i) {
            check(std::isfinite(shared[i]) && std::isfinite(replayed[i]), "finite logits");
            maximum_difference = std::max(maximum_difference, std::abs(shared[i] - replayed[i]));
        }
        check(maximum_difference < 0.02f, "shared KV logits must match replay");
        check(msgl_remove_seq(h.get(), 2) == 0, "replay release");
        const int32_t alternate_token = (tokens.back() + 1) % vocab;
        batch.clear();
        for (size_t i = 0; i < count; ++i)
            batch.push_back({i + 1 == count ? alternate_token : tokens[i], static_cast<int32_t>(i),
                             5, i + 1 == count ? 1 : 0});
        std::vector<float> alternate(vocab);
        check(msgl_forward(h.get(), batch.data(), batch.size(), alternate.data(),
                           alternate.size()) == 0,
              "second prompt reference");
        check(msgl_remove_seq(h.get(), 5) == 0, "reference release");
        batch.clear();
        for (size_t i = 0; i < count; ++i) {
            batch.push_back({tokens[i], static_cast<int32_t>(i), 3, i + 1 == count ? 1 : 0});
            batch.push_back({i + 1 == count ? alternate_token : tokens[i], static_cast<int32_t>(i),
                             4, i + 1 == count ? 1 : 0});
        }
        std::vector<float> parallel(static_cast<size_t>(vocab) * 2);
        check(msgl_forward(h.get(), batch.data(), batch.size(), parallel.data(), parallel.size()) ==
                  0,
              "interleaved two-sequence batch");
        for (int32_t i = 0; i < vocab; ++i) {
            check(std::abs(parallel[i] - initial[i]) < 0.02f, "first compacted logits row");
            check(std::abs(parallel[static_cast<size_t>(vocab) + i] - alternate[i]) < 0.02f,
                  "second compacted logits row");
        }
        check(msgl_remove_seq(h.get(), 3) == 0 && msgl_remove_seq(h.get(), 4) == 0,
              "batch release");
        msgl_chat_message chat{"user", "Hello"};
        size_t chat_size = 0;
        const int32_t template_status =
            msgl_apply_chat_template(h.get(), &chat, 1, 1, nullptr, 0, &chat_size);
        if (template_status >= 0) {
            std::vector<char> rendered(chat_size);
            check(chat_size > 5 && msgl_apply_chat_template(h.get(), &chat, 1, 1, rendered.data(),
                                                            rendered.size(), &chat_size) == 0,
                  "GGUF chat template");
        } else {
            check(std::strstr(msgl_error(), "no chat template") != nullptr,
                  "chat failure must explain missing template");
        }
        std::cout << "ABI model smoke passed: prompt_tokens=" << count << " vocab=" << vocab
                  << " max_logit_difference=" << maximum_difference << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
