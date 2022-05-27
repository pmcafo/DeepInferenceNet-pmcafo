#ifndef __MIEMIEDET_PPYOLOE_HEAD_H__
#define __MIEMIEDET_PPYOLOE_HEAD_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class ConvBNLayer;

class ESEAttn : public Layer
{
public:
    ES