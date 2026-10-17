"""scDiffusion (Luo et al. 2024) driven as a library, the same way CFGen is driven here.

Their CLI scripts need mpi4py/blobfile for distributed training and checkpoint IO; the model
classes themselves do not, so we import those and write our own short training loop.

Their pipeline, kept intact:
  1. a Gaussian VAE on log1p-CPM expression            (VAE/VAE_model.py)
  2. an unconditional diffusion on the VAE latents     (Cell_Unet + create_gaussian_diffusion)
  3. a noisy-latent classifier                          (Cell_classifier)
  4. classifier-GUIDED sampling                         (p_sample_loop with cond_fn)
Conditioning is therefore through a classifier gradient, not a label embedding (CFGen) and not
a decoder covariate (scVI). Three different conditioning mechanisms, same shared decoder.
"""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/Users/fperaltacastro/work/scDiffusion')
from VAE.VAE_model import VAE
from guided_diffusion.cell_model import Cell_Unet, Cell_classifier
from guided_diffusion.script_util import create_gaussian_diffusion

def build_sdiff(Xn_tr, ytr, ncat, seed, ep_vae=40, ep_dif=60, ep_clf=30, bs=256, hid=128, steps=1000,
                vae_steps=None, vae_ckpt=None):
    torch.manual_seed(seed); g=np.random.default_rng(seed)
    G=Xn_tr.shape[1]; X=torch.from_numpy(Xn_tr); Y=torch.from_numpy(ytr)

    vae=VAE(num_genes=G, device='cpu', seed=seed, loss_ae='gauss',
            decoder_activation='linear', hidden_dim=hid)
    # vae_ckpt reuses a converged autoencoder; vae_steps trains to a step budget with THEIR
    # own optimizer (AdamW, lr 5e-4, wd 0.01) at their batch size, because the 40-epoch budget
    # left the round trip unusable (see vae_conv.py / recon_check.py).
    import os
    if vae_ckpt and os.path.exists(vae_ckpt):
        vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'))
    elif vae_steps:
        opt=vae.optimizer_autoencoder
        for _ in range(vae_steps):
            b=X[torch.from_numpy(g.choice(len(X),min(128,len(X)),replace=False))]
            opt.zero_grad(); F.mse_loss(vae(b), b).backward(); opt.step()
    else:
        opt=torch.optim.Adam(vae.parameters(), lr=5e-4, weight_decay=1e-2)
        for _ in range(ep_vae):
            for i in range(0,len(X),bs):
                b=X[torch.from_numpy(g.choice(len(X),min(bs,len(X)),replace=False))]
                opt.zero_grad(); F.mse_loss(vae(b), b).backward(); opt.step()
    vae.eval()
    with torch.no_grad():
        Z=torch.cat([vae(X[i:i+1024], return_latent=True) for i in range(0,len(X),1024)])
    zmu, zsd = Z.mean(0, keepdim=True), Z.std(0, keepdim=True).clamp_min(1e-4)
    Zs=((Z-zmu)/zsd).float()                       # scDiffusion standardises latents before diffusion
    LAT=Zs.shape[1]

    diff=create_gaussian_diffusion(steps=steps, noise_schedule='linear', predict_xstart=False)
    unet=Cell_Unet(input_dim=LAT, hidden_num=[512,256,128,128], dropout=0.1)
    opt=torch.optim.Adam(unet.parameters(), lr=1e-4)
    for _ in range(ep_dif):
        p=g.permutation(len(Zs))
        for i in range(0,len(p),bs):
            b=Zs[torch.from_numpy(p[i:i+bs])]
            if len(b)<2: continue
            t=torch.from_numpy(g.integers(0,steps,len(b)))
            opt.zero_grad(); diff.training_losses(unet,b,t)['loss'].mean().backward(); opt.step()
    unet.eval()

    clf=Cell_classifier(input_dim=LAT, hidden_num=[512,256,128,64], num_class=ncat, dropout=0.1)
    opt=torch.optim.Adam(clf.parameters(), lr=1e-4)
    for _ in range(ep_clf):
        p=g.permutation(len(Zs))
        for i in range(0,len(p),bs):
            j=torch.from_numpy(p[i:i+bs])
            if len(j)<2: continue
            b=Zs[j]; t=torch.from_numpy(g.integers(0,steps,len(j)))
            opt.zero_grad()
            F.cross_entropy(clf(diff.q_sample(b,t), t), Y[j]).backward(); opt.step()
    clf.eval()
    return dict(vae=vae, diff=diff, unet=unet, clf=clf, zmu=zmu, zsd=zsd, lat=LAT)

def sample_sdiff(M, cls, k, scale=10.0, bs=128):
    """Classifier-guided sampling, their cond_fn, returns log1p-CPM expression."""
    def cond_fn(x, t, y=None, **kw):
        with torch.enable_grad():
            xi=x.detach().requires_grad_(True)
            lp=F.log_softmax(M['clf'](xi,t), dim=-1)[range(len(xi)), y.view(-1)]
            return torch.autograd.grad(lp.sum(), xi)[0]*scale
    out=[]
    for i in range(0,k,bs):
        b=min(bs,k-i)
        r=M['diff'].p_sample_loop(M['unet'], (b,M['lat']), clip_denoised=False,
              model_kwargs={'y':torch.full((b,),int(cls),dtype=torch.long)},
              cond_fn=cond_fn, device='cpu', progress=False)
        z=(r[0] if isinstance(r,(tuple,list)) else r).detach()
        with torch.no_grad():
            out.append(M['vae'](z*M['zsd']+M['zmu'], return_decoded=True).numpy())
    # decoder output is only ReLU-clipped (values up to ~65, rows far from 1e4 CPM); put it through the
    # shared normalizer like every other arm: expm1 -> rows to 1e4 -> log1p
    from stat16 import renorm
    return renorm(np.vstack(out))
