#include "loina_internal.h"
#include <stdlib.h>
#include <string.h>

static bool cache_paths(loina_engine *e, loina_session *s, char *state, char *stamp) {
    if (s->temporary || !e->package || !e->runtime || !loina_name_valid(s->id)) return false;
    char identity[65],key[65],config[128],dir[LOINA_PATH],name[160];
    loina_package_identity(e->package,identity);
    loina_sha hash;
    loina_sha_init(&hash);
    loina_sha_update(&hash,identity,65);
    loina_sha_update(&hash,e->runtime->revision,strlen(e->runtime->revision)+1);
    loina_sha_update(&hash,e->chat_identity,strlen(e->chat_identity)+1);
    /* Native KV and rendered prompts must match the saved token stream. */
    int n=snprintf(config,sizeof(config),"loina-%d-chat8:ctx-%u:backend-%u:k-q8_0:v-q4_0",LOINA_ABI,e->config.context,e->selected_backend);
    loina_sha_update(&hash,config,(size_t)n);
    loina_sha_final(&hash,key);
    if (!loina_path(dir,e->cache_directory,key,NULL) || !loina_mkdir(dir)) return false;
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
    transcript_hash(s,text_hash);
    strcpy(e->cache_session,s->id);strcpy(e->cache_transcript,text_hash);
    if (s->manual_save || !e->runtime->save || !cache_paths(e,s,state,stamp)) return false;
    if (snprintf(tmp,sizeof(tmp),"%s.tmp",state)>=(int)sizeof(tmp) ||
        snprintf(meta,sizeof(meta),"%s.tmp",stamp)>=(int)sizeof(meta)) return false;
    if (!e->runtime->save(e->model,tmp) || loina_hash_file(tmp,state_hash,NULL,NULL)) { remove(tmp);return false; }
    FILE *f=fopen(meta,"wb");
    bool ok=f && fprintf(f,"%s\n%s\n",text_hash,state_hash)==130;
    if (f && fclose(f)) ok=false;
    if (ok) ok=loina_replace(tmp,state) && loina_replace(meta,stamp);
    if (!ok) { remove(tmp);remove(meta); }
    return ok;
}
bool loina_cache_restore(loina_engine *e, loina_session *s) {
    char state[LOINA_PATH],stamp[LOINA_PATH],have[131]={0},want[65],hash[65];
    transcript_hash(s,want);
    /* The live context already has this transcript. Restoring it from disk
     * every turn needlessly rewrites GPU KV buffers and discards their layout. */
    if (!strcmp(e->cache_session,s->id) && !strcmp(e->cache_transcript,want)) return true;
    e->cache_session[0]=e->cache_transcript[0]=0;
    if (e->runtime->reset) e->runtime->reset(e->model);
    if (!e->runtime->restore || !cache_paths(e,s,state,stamp)) return false;
    FILE *f=fopen(stamp,"rb");
    if (!f) return false;
    size_t n=fread(have,1,sizeof(have),f);
    fclose(f);
    if (n!=130 || have[64]!='\n' || have[129]!='\n' || memcmp(have,want,64)) return false;
    if (loina_hash_file(state,hash,NULL,NULL) || memcmp(hash,have+65,64)) return false;
    if (!e->runtime->restore(e->model,state)) return false;
    strcpy(e->cache_session,s->id);strcpy(e->cache_transcript,want);
    return true;
}

loina_status loina_session_checkpoint(loina_session *s, loina_error *error) {
    if (!s || s->temporary) return loina_fail(error,LOINA_INVALID,"Session cannot be persisted");
    loina_engine *e=s->engine;
    if (atomic_exchange(&e->busy,true)) return loina_fail(error,LOINA_BUSY,"Engine is busy");
    loina_status status=LOINA_OK;
    char hash[65];
    transcript_hash(s,hash);
    if (e->model && e->runtime && (strcmp(e->cache_session,s->id) || strcmp(e->cache_transcript,hash)))
        loina_cache_restore(e,s);
    if (!e->model || !e->runtime || !e->runtime->save || strcmp(e->cache_session,s->id) ||
        strcmp(e->cache_transcript,hash)) {
        status=loina_fail(error,LOINA_INVALID,"No current KV checkpoint; complete a chat turn before saving");
    } else {
        status=loina_session_save(s,error);
        if (!status) {
            bool manual=s->manual_save;
            s->manual_save=false;
            if (!loina_cache_save(e,s)) status=loina_fail(error,LOINA_IO,"Cannot save KV checkpoint");
            s->manual_save=manual;
        }
    }
    atomic_store(&e->busy,false);
    return status;
}
