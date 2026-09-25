// Local, CPU-only Python/Go EHBP interoperability fixture. No production calls.
package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/tinfoilsh/encrypted-http-body-protocol/identity"
)

const model = "lebrel/deepseek-v4-flash-uncensored"

func main() {
	id, err := identity.NewIdentity()
	if err != nil {
		panic(err)
	}
	config, _ := id.MarshalConfig()
	digest := sha256.Sum256(config)
	keyID := hex.EncodeToString(digest[:])
	// Publicly known test seed, never a production secret.
	key := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{7}, 32))
	pub := key.Public().(ed25519.PublicKey)
	signer := sha256.Sum256(pub)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/lebrel-encryption", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			http.Error(w, "metadata received credential", 400)
			return
		}
		payload, _ := json.Marshal(map[string]any{"version": 1, "issuedAt": time.Now().Unix(), "expiresAt": time.Now().Unix() + 300, "hpkeConfig": base64.StdEncoding.EncodeToString(config), "keyId": keyID, "modelId": model, "serverInstanceId": strings.Repeat("0", 32), "signingKeyId": hex.EncodeToString(signer[:])})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"version": 1, "payload": base64.StdEncoding.EncodeToString(payload), "signature": base64.StdEncoding.EncodeToString(ed25519.Sign(key, payload))})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer leb_live_test" || len(r.Header.Get("X-Lebrel-Request-Id")) != 36 || r.Header.Get("X-Lebrel-Encryption-Key-Id") != keyID {
			http.Error(w, "missing required headers", 400)
			return
		}
		wire, _ := io.ReadAll(r.Body)
		if bytes.Contains(wire, []byte("secret-user-text")) || bytes.Contains(wire, []byte(model)) {
			http.Error(w, "plaintext detected", 400)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(wire))
		ctx, err := id.DecryptRequestWithContext(r)
		if err != nil {
			http.Error(w, "decrypt setup", 400)
			return
		}
		plain, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "decrypt failed", 400)
			return
		}
		var request struct {
			Model    string `json:"model"`
			Stream   bool   `json:"stream"`
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(plain, &request) != nil || request.Model != model || len(request.Messages) != 1 {
			http.Error(w, "invalid request", 400)
			return
		}
		mode := request.Messages[0].Content
		if mode == "plain-success" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"choices":[{"finish_reason":"stop"}]}`)
			return
		}
		if mode == "plain-error" {
			http.Error(w, "secret-user-text", 422)
			return
		}
		var target http.ResponseWriter = w
		if mode == "tamper" || mode == "truncated-frame" {
			recorder := httptest.NewRecorder()
			target = recorder
			defer func() {
				for name, values := range recorder.Header() {
					for _, value := range values {
						w.Header().Add(name, value)
					}
				}
				wire := recorder.Body.Bytes()
				if mode == "tamper" {
					wire[len(wire)-1] ^= 1
				} else {
					wire = wire[:len(wire)-1]
				}
				w.WriteHeader(recorder.Code)
				_, _ = w.Write(wire)
			}()
		}
		ew, err := id.SetupDerivedResponseEncryption(target, ctx)
		if err != nil {
			panic(err)
		}
		ew.Header().Set("X-Lebrel-Encryption-Key-Id", keyID)
		if mode == "encrypted-error" {
			ew.Header().Set("Content-Type", "application/json")
			ew.WriteHeader(402)
			_, _ = io.WriteString(ew, `{"error":{"code":"insufficient_credits","message":"secret-user-text"}}`)
			return
		}
		if request.Stream {
			ew.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(ew, "data: {\"id\":\"local\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"verified-python-go 🐺\"},\"finish_reason\":null}]}\n\n")
			ew.Flush()
			if mode == "blocking" {
				<-r.Context().Done()
				return
			}
			if mode != "done-no-finish" {
				_, _ = io.WriteString(ew, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
				ew.Flush()
			}
			if mode != "incomplete" {
				_, _ = io.WriteString(ew, "data: [DONE]\n\n")
				ew.Flush()
			}
			if mode == "after-done" {
				_, _ = io.WriteString(ew, "data: {\"choices\":[]}\n\n")
				ew.Flush()
			}
			return
		}
		ew.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(ew, `{"id":"local","model":"`+model+`","choices":[{"index":0,"message":{"role":"assistant","content":"verified-python-go"},"finish_reason":"stop"}]}`)
	})
	fmt.Printf("%s %s\n", "http://"+listener.Addr().String(), base64.StdEncoding.EncodeToString(pub))
	if err := http.Serve(listener, mux); err != nil {
		panic(err)
	}
}
