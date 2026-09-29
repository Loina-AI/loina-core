#include "loina_internal.h"
#include <stdlib.h>
#include <string.h>

static bool cache_paths(loina_engine *e, loina_session *s, char *state, char *stamp) {
    if (!e->package || !e->runtime || !loina_name_valid(s->id)) return false;
    char identity[65],key[65],config[128],dir[LOINA_PATH],name[160];
    loina_package_identity(e->package,identity);
    loina_sha hash;
    loina_sha_init(&hash);
    loina_sha_update(&hash,identity,65);
    loina_sha_update(&hash,e->runtime->revision,strlen(e->runtime->revision)+1);
    /* Native KV and rendered prompts must match the saved token stream. */
    int n=snprintf(config,sizeof(config),"loina-%d-chat2:ctx-%u:k-q8_0:v-q4_0",LOINA_ABI,e->config.context);
    loina_sha_update(&hash,config,(size_t)n);
    loina_sha_final(&hash,key);
    if (!loina_path(dir,e->directory,"cache",key) || !loina_mkdir(dir)) return false;
    snprintf(name,sizeof(name),"%s.kv",s->id);
    if (!loina_path(state,dir,name,NULL)) return false;
    snprintf(name,sizeof(name),"%s.stamp",s->id);
    return loina_path(stamp,dir,name,NULL);
}
static void transcript_hash(loina_session *s, char hex[65]) {
    loina_sha hash;
    loina_sha_init(&hash);
    char first[32];
    int n=snprintf(first,sizeof(first),"%zu:",s->first);
    loina_sha_update(&hash,first,(size_t)n);
    loina_sha_update(&hash,s->package_id,strlen(s->package_id)+1);
    for (size_t i=0;i<s->count;i++) {
        unsigned char tags[2]={(unsigned char)s->messages[i].role,(unsigned char)s->messages[i].kind};
        loina_sha_update(&hash,tags,2);
        loina_sha_update(&hash,s->messages[i].text,strlen(s->messages[i].text)+1);
    }
    loina_sha_final(&hash,hex);
}
bool loina_cache_save(loina_engine *e, loina_session *s) {
    char state[LOINA_PATH],stamp[LOINA_PATH],tmp[LOINA_PATH],meta[LOINA_PATH],text_hash[65],state_hash[65];
    if (!e->runtime->save || !cache_paths(e,s,state,stamp)) return false;
    if (snprintf(tmp,sizeof(tmp),"%s.tmp",state)>=(int)sizeof(tmp) ||
        snprintf(meta,sizeof(meta),"%s.tmp",stamp)>=(int)sizeof(meta)) return false;
    if (!e->runtime->save(e->model,tmp) || loina_hash_file(tmp,state_hash,NULL,NULL)) { remove(tmp);return false; }
    transcript_hash(s,text_hash);
    FILE *f=fopen(meta,"wb");
    bool ok=f && fprintf(f,"%s\n%s\n",text_hash,state_hash)==130;
    if (f && fclose(f)) ok=false;
    if (ok) ok=loina_replace(tmp,state) && loina_replace(meta,stamp);
    if (!ok) { remove(tmp);remove(meta); }
    return ok;
}
bool loina_cache_restore(loina_engine *e, loina_session *s) {
    char state[LOINA_PATH],stamp[LOINA_PATH],have[131]={0},want[65],hash[65];
    if (!e->runtime->restore || !cache_paths(e,s,state,stamp)) return false;
    FILE *f=fopen(stamp,"rb");
    if (!f) return false;
    size_t n=fread(have,1,sizeof(have),f);
    fclose(f);
    transcript_hash(s,want);
    if (n!=130 || have[64]!='\n' || have[129]!='\n' || memcmp(have,want,64)) return false;
    if (loina_hash_file(state,hash,NULL,NULL) || memcmp(hash,have+65,64)) return false;
    return e->runtime->restore(e->model,state);
}
