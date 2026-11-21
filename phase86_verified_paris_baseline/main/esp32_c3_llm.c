// Phase 85: fixed-prompt prefill plus one generated token Qwen2.5-0.5B-Instruct INT8/W8A32 on ESP32-C3.
// PC transfers immutable weights/embedding; all matvecs, RMSNorm, MLP, residual and argmax run here.
// Full causal GQA attention and RoPE, bounded KV cache, all on ESP32.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "driver/usb_serial_jtag.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_err.h"
#define D 896
#define FF 4864
#define KV 128
#define VOCAB 151936
#define EPS 1e-6f
#define POSITIONS 6
#define LAYERS 24
static float x[D], norm_in[D], norm_post[D], layer_norm[D], q[D], k[KV], v[KV];
static float proj[D], gate[FF], up[FF], act[FF], bias[D];
static float kv_key[LAYERS][POSITIONS][KV], kv_value[LAYERS][POSITIONS][KV];
static unsigned position=0;
static unsigned selected_token=0;
static unsigned char row[FF+4];
static char textline[130];
static bool tx(const char *s){size_t n=strlen(s),i=0;while(i<n){int r=usb_serial_jtag_write_bytes(s+i,n-i,pdMS_TO_TICKS(4000));if(r<=0)return false;i+=(size_t)r;}return usb_serial_jtag_wait_tx_done(pdMS_TO_TICKS(5000))==ESP_OK;}
static bool rx(void *p,size_t n){uint8_t *a=p;size_t i=0;int64_t lim=esp_timer_get_time()+30000000LL;while(i<n){int r=usb_serial_jtag_read_bytes(a+i,n-i,pdMS_TO_TICKS(200));if(r>0){i+=r;lim=esp_timer_get_time()+30000000LL;}else if(esp_timer_get_time()>lim)return false;}return true;}
static bool rxline(char *s,size_t cap){size_t i=0;char c;while(i+1<cap){if(!rx(&c,1))return false;if(c=='\n'){s[i]=0;return true;}if(c!='\r')s[i++]=c;}return false;}
static bool load_param(int li,const char *type,const char *name,float *dst,unsigned n){snprintf(textline,sizeof(textline),"PARAM %d %s %s %u\n",li,type,name,n*4);return tx(textline)&&rx(dst,n*4);}
// Device-driven bounded streaming: host never sends an entire matrix blindly.
// Every requested chunk is <= 4096 bytes, below the 8192-byte USB RX ring.
static bool request_chunk(unsigned bytes){
    snprintf(textline,sizeof(textline),"CHUNK %u\n",bytes);
    return tx(textline);
}
static bool matrix(int li,const char *name,const float *in,float *dst,unsigned rows,unsigned cols){
    snprintf(textline,sizeof(textline),"MATRIX %d %s %u %u\n",li,name,rows,cols);
    if(!tx(textline))return false;
    const unsigned stride=cols+4;
    const unsigned batch=(stride>4096)?1:(4096/stride);
    for(unsigned i=0;i<rows;){
        unsigned count=rows-i;
        if(count>batch)count=batch;
        if(!request_chunk(count*stride))return false;
        for(unsigned j=0;j<count;j++){
            if(!rx(row,stride))return false;
            float sc;memcpy(&sc,row+cols,4);
            if(!isfinite(sc)||sc<=0)return false;
            const int8_t *w=(const int8_t *)row;
            float sum=0.f;
            for(unsigned k=0;k<cols;k++)sum+=(float)w[k]*in[k];
            dst[i+j]=sum*sc;
        }
        i+=count;
    }
    return true;
}
static void rms(const float *a,const float *weight,float *out,unsigned n){
    float sum=0.f;for(unsigned i=0;i<n;i++)sum+=a[i]*a[i];
    float inv=1.f/sqrtf(sum/n+EPS);
    for(unsigned i=0;i<n;i++)out[i]=a[i]*inv*weight[i];
}
static bool send_state(int li){
    snprintf(textline,sizeof(textline),"STATE %d %u\n",li,D*4);
    if(!tx(textline)||!rxline(textline,sizeof(textline))||strcmp(textline,"SEND")!=0)return false;
    const uint8_t *ptr=(const uint8_t*)x;
    for(unsigned i=0;i<D*4;){unsigned n=D*4-i;if(n>256)n=256;int r=usb_serial_jtag_write_bytes((const char*)ptr+i,n,pdMS_TO_TICKS(4000));if(r<=0)return false;i+=r;}
    if(usb_serial_jtag_wait_tx_done(pdMS_TO_TICKS(5000))!=ESP_OK)return false;
    return rxline(textline,sizeof(textline))&&strcmp(textline,"GOT")==0;
}
static bool one_layer(int li){
    if(!load_param(li,"norm","input_layernorm",norm_in,D))return false;
    if(!load_param(li,"norm","post_attention_layernorm",norm_post,D))return false;
    rms(x,norm_in,layer_norm,D);
    if(!matrix(li,"q_proj",layer_norm,q,D,D))return false;
    if(!load_param(li,"bias","q_proj",bias,D))return false;
    for(unsigned i=0;i<D;i++)q[i]+=bias[i];
    if(!matrix(li,"k_proj",layer_norm,k,KV,D))return false;
    if(!load_param(li,"bias","k_proj",bias,KV))return false;
    for(unsigned i=0;i<KV;i++)k[i]+=bias[i];
    if(!matrix(li,"v_proj",layer_norm,v,KV,D))return false;
    if(!load_param(li,"bias","v_proj",bias,KV))return false;
    for(unsigned i=0;i<KV;i++)v[i]+=bias[i];
    // Qwen2.5 RoPE with 64-wide heads, theta=1,000,000, rotate_half layout.
    // The ESP rotates Q and K for the current position before caching the key.
    if(position>0){
        for(unsigned j=0;j<32;j++){
            float angle=(float)position*powf(1000000.0f,-(float)(2*j)/64.0f);
            float c=cosf(angle), si=sinf(angle);
            for(unsigned h=0;h<14;h++){
                unsigned a=h*64+j,b=a+32;
                float u=q[a],w=q[b];q[a]=u*c-w*si;q[b]=w*c+u*si;
            }
            for(unsigned h=0;h<2;h++){
                unsigned a=h*64+j,b=a+32;
                float u=k[a],w=k[b];k[a]=u*c-w*si;k[b]=w*c+u*si;
            }
        }
    }
    memcpy(kv_key[li][position],k,sizeof(k));
    memcpy(kv_value[li][position],v,sizeof(v));
    if(position==0){
        // A one-element causal softmax has weight exactly 1.
        for(unsigned h=0;h<14;h++)for(unsigned j=0;j<64;j++)q[h*64+j]=v[(h/7)*64+j];
    }else{
        // Softmax across all cached positions, including the current position.
        for(unsigned h=0;h<14;h++){
            unsigned kh=h/7;
            float scores[POSITIONS], peak=-INFINITY, denom=0.f;
            for(unsigned t=0;t<=position;t++){
                float score=0.f;
                for(unsigned j=0;j<64;j++)score+=q[h*64+j]*kv_key[li][t][kh*64+j];
                scores[t]=score*0.125f;
                if(scores[t]>peak)peak=scores[t];
            }
            for(unsigned t=0;t<=position;t++){
                scores[t]=expf(scores[t]-peak);
                denom+=scores[t];
            }
            if(!isfinite(denom)||denom<=0.f)return false;
            for(unsigned j=0;j<64;j++){
                float value=0.f;
                for(unsigned t=0;t<=position;t++)
                    value+=(scores[t]/denom)*kv_value[li][t][kh*64+j];
                q[h*64+j]=value;
            }
        }
    }
    if(!matrix(li,"o_proj",q,proj,D,D))return false;
    for(unsigned i=0;i<D;i++)x[i]+=proj[i];
    rms(x,norm_post,layer_norm,D);
    if(!matrix(li,"gate_proj",layer_norm,gate,FF,D))return false;
    if(!matrix(li,"up_proj",layer_norm,up,FF,D))return false;
    for(unsigned i=0;i<FF;i++)act[i]=(gate[i]/(1.f+expf(-gate[i])))*up[i];
    if(!matrix(li,"down_proj",act,proj,D,FF))return false;
    for(unsigned i=0;i<D;i++)x[i]+=proj[i];
    return send_state(li);
}
static bool output_head(unsigned *out_id,float *out_score){
    if(!tx("HEAD_NORM 3584\n")||!rx(norm_in,D*4))return false;
    rms(x,norm_in,layer_norm,D);
    if(!tx("HEAD_ROWS 151936 1792\n"))return false;
    float best=-INFINITY;unsigned arg=0;
    for(unsigned i=0;i<VOCAB;){
        unsigned count=VOCAB-i;if(count>4)count=4;
        if(!request_chunk(count*D*2))return false;
        for(unsigned r=0;r<count;r++){
            if(!rx(row,D*2))return false;
            float score=0.f;
            for(unsigned j=0;j<D;j++){
                uint16_t b=(uint16_t)row[2*j]|((uint16_t)row[2*j+1]<<8);
                uint32_t bits=(uint32_t)b<<16;float w;memcpy(&w,&bits,4);
                score+=w*layer_norm[j];
            }
            if(score>best){best=score;arg=i+r;}
        }
        i+=count;
    }
    *out_id=arg;*out_score=best;
    snprintf(textline,sizeof(textline),"TOKEN %u %.9g\n",arg,(double)best);
    return tx(textline);
}
static void run_prompt(unsigned count){
    if(count<1 || count>POSITIONS){tx("ERR PROMPT_LENGTH\n");return;}
    int64_t begin=esp_timer_get_time();
    for(position=0;position<count;position++){
        // PC supplies only pretrained embedding for the host-tokenized prompt ID.
        snprintf(textline,sizeof(textline),"EMBED_POS %u 3584\n",position);
        if(!tx(textline)||!rx(x,D*4)){tx("ERR EMBED\n");return;}
        for(unsigned li=0;li<LAYERS;li++){
            if(!one_layer((int)li)){
                snprintf(textline,sizeof(textline),"ERR LAYER %u POSITION %u\n",li+1,position);
                tx(textline);return;
            }
            snprintf(textline,sizeof(textline),"LAYER_DONE %u %u\n",position,li+1);
            if(!tx(textline))return;
        }
    }
    // Only the last prompt position needs a vocabulary projection.
    // Argmax, RMSNorm and BF16 head dot products execute on the ESP32.
    float score;unsigned token;
    if(!output_head(&token,&score)){tx("ERR HEAD\n");return;}
    selected_token=token;
    snprintf(textline,sizeof(textline),"DONE P85 positions=%u device_us=%lld heap=%u\n",
        count,(long long)(esp_timer_get_time()-begin),
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT));
    tx(textline);
}

void app_main(void){
    usb_serial_jtag_driver_config_t conf={.rx_buffer_size=8192,.tx_buffer_size=2048};
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&conf));
    tx("READY PHASE85 W8A32\n");
    while(true){
        if(!rxline(textline,sizeof(textline)))continue;
        unsigned count=0; char tail=0;
        if(sscanf(textline,"STARTP %u %c",&count,&tail)==1)run_prompt(count);
        else if(strcmp(textline,"PING")==0)tx("PONG P85\n");
        else tx("ERR COMMAND\n");
    }
}
