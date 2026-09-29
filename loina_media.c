#include "loina_internal.h"
#include <stdlib.h>
#include <string.h>

static void be32(unsigned char *p, uint32_t x) {
    p[0]=(unsigned char)(x>>24);p[1]=(unsigned char)(x>>16);p[2]=(unsigned char)(x>>8);p[3]=(unsigned char)x;
}
static uint32_t crc32(const unsigned char *p, size_t n, uint32_t crc) {
    while (n--) {
        crc^=*p++;
        for (int i=0;i<8;i++) crc=(crc>>1)^(0xedb88320u & (0u-(crc&1u)));
    }
    return crc;
}
static bool png_chunk(FILE *f, const char type[4], const unsigned char *data, uint32_t n) {
    unsigned char len[4],checksum[4];
    be32(len,n);
    uint32_t crc=crc32((const unsigned char *)type,4,UINT32_MAX);
    be32(checksum,crc32(data,n,crc)^UINT32_MAX);
    return fwrite(len,1,4,f)==4 && fwrite(type,1,4,f)==4 &&
        fwrite(data,1,n,f)==n && fwrite(checksum,1,4,f)==4;
}
loina_status loina_png(const char *path, const unsigned char *pixels, int w, int h, int channels, loina_error *error) {
    if (!pixels || w<1 || h<1 || w>4096 || h>4096 || (channels!=3 && channels!=4))
        return loina_fail(error,LOINA_INVALID,"Invalid generated image");
    size_t stride=(size_t)w*(size_t)channels;
    size_t raw_size=(stride+1)*(size_t)h;
    unsigned char *raw=malloc(raw_size),*zip=malloc(raw_size+raw_size/65535*5+16);
    if (!raw || !zip) { free(raw);free(zip);return loina_fail(error,LOINA_MEMORY,"Cannot encode image"); }
    for (int y=0;y<h;y++) { raw[(stride+1)*(size_t)y]=0;memcpy(raw+(stride+1)*(size_t)y+1,pixels+stride*(size_t)y,stride); }
    /* Stored DEFLATE blocks keep the PNG writer small and dependency-free.
     * These images are small; inference dominates both memory and latency. */
    size_t used=2,offset=0;
    zip[0]=0x78;zip[1]=0x01;
    while (offset<raw_size) {
        size_t n=raw_size-offset;
        if (n>65535) n=65535;
        zip[used++]=(unsigned char)(offset+n==raw_size);
        zip[used++]=(unsigned char)n;zip[used++]=(unsigned char)(n>>8);
        zip[used++]=(unsigned char)~n;zip[used++]=(unsigned char)(~n>>8);
        memcpy(zip+used,raw+offset,n);used+=n;offset+=n;
    }
    uint32_t a=1,b=0;
    for (size_t i=0;i<raw_size;i++) { a=(a+raw[i])%65521;b=(b+a)%65521; }
    be32(zip+used,(b<<16)|a);used+=4;
    free(raw);
    unsigned char header[13]={0};
    be32(header,(uint32_t)w);be32(header+4,(uint32_t)h);header[8]=8;header[9]=channels==3 ? 2 : 6;
    char temp[LOINA_PATH];
    if (!path || snprintf(temp,sizeof(temp),"%s.tmp",path)>=(int)sizeof(temp)) { free(zip);return loina_fail(error,LOINA_INVALID,"Image output path is too long"); }
    FILE *f=fopen(temp,"wb");
    bool ok=f && fwrite("\211PNG\r\n\032\n",1,8,f)==8 && png_chunk(f,"IHDR",header,13) &&
        png_chunk(f,"IDAT",zip,(uint32_t)used) && png_chunk(f,"IEND",NULL,0);
    free(zip);
    if (f && fclose(f)) ok=false;
    if (ok) ok=loina_replace(temp,path);
    if (!ok) { remove(temp);return loina_fail(error,LOINA_IO,"Cannot write generated PNG"); }
    return LOINA_OK;
}
static uint32_t le32(const unsigned char *p) { return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
static uint16_t le16(const unsigned char *p) { return (uint16_t)((unsigned)p[0] | (unsigned)p[1]<<8); }
loina_status loina_wav(const char *path, float **samples, int *count, loina_error *error) {
    *samples=NULL;*count=0;
    uint64_t size=loina_file_bytes(path);
    if (size<44 || size>16000u*2u*300u+4096u) return loina_fail(error,LOINA_INVALID,"Audio must be a PCM16 mono 16kHz WAV, at most five minutes");
    FILE *f=fopen(path,"rb");
    if (!f) return loina_fail(error,LOINA_IO,"Cannot read audio file");
    unsigned char header[12],chunk[8],fmt[16];
    bool valid=fread(header,1,12,f)==12 && !memcmp(header,"RIFF",4) && !memcmp(header+8,"WAVE",4) &&
        (uint64_t)le32(header+4)+8<=size;
    bool formatted=false;
    uint32_t bytes=0;
    long data_offset=0;
    while (valid && fread(chunk,1,8,f)==8) {
        uint32_t n=le32(chunk+4);
        long start=ftell(f);
        if (start<0 || (uint64_t)start+n>size || n>size) { valid=false;break; }
        if (!memcmp(chunk,"fmt ",4)) {
            if (n<16 || fread(fmt,1,16,f)!=16) { valid=false;break; }
            formatted=le16(fmt)==1 && le16(fmt+2)==1 && le32(fmt+4)==16000 &&
                le32(fmt+8)==32000 && le16(fmt+12)==2 && le16(fmt+14)==16;
        } else if (!memcmp(chunk,"data",4)) { bytes=n;data_offset=start; }
        if (fseek(f,start+(long)n+(long)(n&1u),SEEK_SET)) { valid=false;break; }
    }
    if (!valid || !formatted || !bytes || bytes%2 || bytes>16000u*2u*300u || !data_offset) {
        fclose(f);return loina_fail(error,LOINA_INVALID,"Unsupported or corrupt audio; expected PCM16 mono 16kHz WAV");
    }
    float *out=malloc((size_t)(bytes/2)*sizeof(*out));
    if (!out) { fclose(f);return loina_fail(error,LOINA_MEMORY,"Cannot allocate audio samples"); }
    bool ok=fseek(f,data_offset,SEEK_SET)==0;
    unsigned char block[4096];
    size_t done=0;
    while (ok && done<bytes) {
        size_t n=bytes-done;
        if (n>sizeof(block)) n=sizeof(block);
        if (fread(block,1,n,f)!=n) { ok=false;break; }
        for (size_t i=0;i<n;i+=2) out[(done+i)/2]=(float)(int16_t)le16(block+i)/32768.0f;
        done+=n;
    }
    fclose(f);
    if (!ok) { free(out);return loina_fail(error,LOINA_IO,"Cannot read audio samples"); }
    *samples=out;*count=(int)(bytes/2);
    return LOINA_OK;
}
